"""P0 索引：扫描原始数据（Kaggle 同格式 parquet）的每个地图目录，生成索引与数据检查报告。

环境变量：CS2_ROOT（原始数据所在目录，默认见 paths.py）、CS2_RAW_GLOB（数据集目录名模式，默认 cs2-s-tier-demos-*）、
         CS2_INDEX（输出目录，默认 $CS2_ROOT/_index）
输出：maps.csv, map_players.csv, events.csv, reports.csv, summary.txt
"""
import paths
import os, re, json, glob, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pyarrow.parquet as pq

B = paths.ROOT
RAW_GLOB = os.environ.get("CS2_RAW_GLOB", "cs2-s-tier-demos-*")
OUT = os.environ.get("CS2_INDEX", f"{B}/_index")
os.makedirs(OUT, exist_ok=True)
TABLES = ["rounds", "ticks", "kills", "damages", "shots", "grenades", "smokes", "infernos", "bomb", "footsteps"]
REQ = {f"{t}.parquet" for t in TABLES} | {"header.json"}


def legal_score(a, b):
    hi, lo = max(a, b), min(a, b)
    if hi == 13 and lo <= 11:
        return True
    if hi < 16 or (hi - 13) % 3:
        return False
    return hi - 4 <= lo <= hi - 2


def scan_map(path):
    r = {"path": path}
    rel = os.path.relpath(path, B).split(os.sep)
    r["dataset"] = rel[0].replace("cs2-s-tier-demos-", "")
    r["depth_ok"] = len(rel) == 5
    if len(rel) == 5:
        r["outer"], r["event_dir"], r["match_id"], r["map_folder"] = rel[1:]
    files = set(os.listdir(path))
    r["missing"] = ",".join(sorted(REQ - files))
    players = []
    try:
        h = json.load(open(f"{path}/header.json"))
        r["patch"] = h.get("patch_version")
        r["map_name"] = h.get("map_name")
    except Exception as e:
        r["err_header"] = str(e)[:80]
    for t in TABLES:
        try:
            r[f"rows_{t}"] = pq.ParquetFile(f"{path}/{t}.parquet").metadata.num_rows
        except Exception:
            r[f"rows_{t}"] = -1
    try:
        rd = pd.read_parquet(f"{path}/rounds.parquet")
        r["n_rounds_raw"] = len(rd)
        r["has_bomb_cols"] = "bomb_plant" in rd.columns
        zero = rd["start"] == rd["end"]
        r["n_zero_rounds"] = int(zero.sum())
        rd = rd[~zero]
        r["n_rounds"] = len(rd)
        r["dup_round_num"] = int(rd["round_num"].duplicated().sum())
        w = rd["winner"].astype(str).str.lower()
        r["bad_winner"] = int((~w.isin(["ct", "t"])).sum())

        tk = pq.read_table(f"{path}/ticks.parquet", columns=["tick", "steamid", "name", "side", "round_num"]).to_pandas()
        ut = tk["tick"].drop_duplicates().sort_values()
        r["n_ticks"] = len(ut)
        r["tick_step"] = float(ut.diff().median()) if len(ut) > 1 else None
        r["rows_per_tick"] = round(len(tk) / max(len(ut), 1), 2)
        r["n_players"] = int(tk["steamid"].nunique())
        sides = tk.drop_duplicates(["round_num", "steamid", "side"])[["round_num", "steamid", "side"]]
        sides = sides[sides["side"].isin(["ct", "t"])]
        # 用第一个 5v5 的回合确定两队
        teamA = teamB = None
        for rn, g in sides.groupby("round_num"):
            ct, t = set(g[g.side == "ct"].steamid), set(g[g.side == "t"].steamid)
            if len(ct) == 5 and len(t) == 5:
                teamA, teamB = ct, t
                break
        r["teams_ok"] = teamA is not None
        if teamA:
            ct_team = {}
            for rn, g in sides[sides.side == "ct"].groupby("round_num"):
                s = set(g.steamid)
                a, b = len(s & teamA), len(s & teamB)
                ct_team[rn] = "A" if a > b else ("B" if b > a else None)
            rd2 = rd.assign(ct_team=rd["round_num"].map(ct_team))
            r["rounds_team_unknown"] = int(rd2["ct_team"].isna().sum())
            win_team = rd2.apply(lambda x: x.ct_team if str(x.winner).lower() == "ct" else ({"A": "B", "B": "A"}.get(x.ct_team)), axis=1)
            r["score_A"], r["score_B"] = int((win_team == "A").sum()), int((win_team == "B").sum())
            r["legal_score"] = legal_score(r["score_A"], r["score_B"])
            names = tk.drop_duplicates("steamid").set_index("steamid")["name"]
            for sid in tk["steamid"].unique():
                team = "A" if sid in teamA else ("B" if sid in teamB else "?")
                players.append({"path": path, "steamid": str(sid), "name": names.get(sid), "team": team})
            # kills 表里 ct_side / t_side 是否是队名
            try:
                k = pd.read_parquet(f"{path}/kills.parquet", columns=["round_num", "ct_side", "t_side"]).drop_duplicates()
                ka = k[k.round_num.map(ct_team) == "A"]
                if len(ka):
                    r["teamA_name"], r["teamB_name"] = str(ka.iloc[0].ct_side), str(ka.iloc[0].t_side)
            except Exception as e:
                r["err_teamname"] = str(e)[:80]
    except Exception as e:
        r["err_scan"] = str(e)[:120]
    return r, players


def tick_sample(path):
    """抽样检查 tick 级数据质量。"""
    cols = ["tick", "steamid", "side", "health", "X", "Y", "Z", "yaw", "pitch", "round_num"]
    tk = pd.read_parquet(f"{path}/ticks.parquet", columns=cols)
    per_tick = tk.groupby("tick").size()
    alive = tk[tk.health > 0].sort_values(["steamid", "tick"])
    d = (alive.groupby("steamid")[["X", "Y"]].diff() ** 2).sum(axis=1) ** 0.5
    return {"path": path, "null_xy": float(tk[["X", "Y"]].isna().any(axis=1).mean()),
            "null_yaw": float(tk["yaw"].isna().mean()),
            "ticks_with_10": float((per_tick == 10).mean()),
            "jump_gt_300_per_1k": float((d > 300).mean() * 1000)}


def year_of(name):
    m = re.search(r"20(2[3-6])", name)
    return int("20" + m.group(1)) if m else None


def main():
    t0 = time.time()
    map_dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{B}/{RAW_GLOB}/**/rounds.parquet", recursive=True)})
    print(f"找到 {len(map_dirs)} 个地图目录，开始扫描……", flush=True)
    rows, players = [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for i, (r, p) in enumerate(ex.map(scan_map, map_dirs, chunksize=4)):
            rows.append(r); players.extend(p)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(map_dirs)}  {time.time() - t0:.0f}s", flush=True)
    maps = pd.DataFrame(rows)
    mp = pd.DataFrame(players)
    for c in ["teamA_name", "teamB_name", "patch", "map_name", "score_A", "score_B", "legal_score", "teams_ok",
              "has_bomb_cols", "n_rounds", "n_zero_rounds", "dup_round_num", "bad_winner", "n_players",
              "tick_step", "rows_per_tick", "n_ticks", "event_dir", "match_id"]:
        if c not in maps:
            maps[c] = None

    # 事件、年份
    maps["event_id"] = maps["event_dir"].str.extract(r"^(E\d+)_")[0]
    maps["event_name"] = maps["event_dir"].str.replace(r"^E\d+_", "", regex=True).str.replace("_", " ")
    maps["match_id_num"] = pd.to_numeric(maps["match_id"], errors="coerce")
    ev = maps.groupby(["event_id", "event_name"]).agg(matches=("match_id", "nunique"), maps=("path", "size"),
                                                      mid_min=("match_id_num", "min"), mid_med=("match_id_num", "median"),
                                                      patch_min=("patch", "min"), patch_max=("patch", "max")).reset_index()
    ev["year_from_name"] = ev["event_name"].map(year_of)
    ev = ev.sort_values("mid_med")
    ev["year"] = ev["year_from_name"].ffill().bfill().fillna(time.localtime().tm_year)   # 都推断不出时用当前年份   # 名字里没有年份的，用比赛 ID 相邻的赛事补
    maps = maps.merge(ev[["event_id", "year"]], on="event_id", how="left")
    ev.to_csv(f"{OUT}/events.csv", index=False)

    # 转换报告（带 dem 文件名 → 对阵队名）
    reps = []
    for f in glob.glob(f"{B}/**/*.csv", recursive=True):
        if "/_index/" in f:
            continue
        try:
            d = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        except Exception:
            continue
        if "dem_name" in d.columns and "match_id" in d.columns:
            d["report_file"] = os.path.relpath(f, B)
            reps.append(d)
    rep = pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()
    rep.to_csv(f"{OUT}/reports.csv", index=False)

    maps.to_csv(f"{OUT}/maps.csv", index=False)
    if len(mp):
        mp = mp.merge(maps[["path", "match_id", "year", "map_name", "event_name", "teamA_name", "teamB_name"]], on="path", how="left")
        mp["team_name"] = mp.apply(lambda x: x.teamA_name if x.team == "A" else (x.teamB_name if x.team == "B" else None), axis=1)
        mp.drop(columns=["teamA_name", "teamB_name"]).to_csv(f"{OUT}/map_players.csv", index=False)

    # tick 抽样
    samp = maps[maps.get("err_scan").isna()] if "err_scan" in maps else maps
    samp = samp.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(len(g), 6), random_state=0))["path"].tolist()
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        ts = pd.DataFrame(list(ex.map(tick_sample, samp)))

    # ====== 汇总 ======
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    ok = maps[maps.depth_ok]
    P("# 数据可行性检查", time.strftime("%F %T"), f"（扫描用时 {time.time() - t0:.0f}s）")
    P("\n## 1. 规模")
    P(f"地图目录 {len(maps)}，目录层级异常 {int((~maps.depth_ok).sum())}")
    P(f"赛事 {ok.event_id.nunique()}，比赛(去重) {ok.match_id.nunique()}，地图 {len(ok)}")
    P(f"有效回合（去掉零时长）{int(ok.n_rounds.sum())}，零时长回合 {int(ok.n_zero_rounds.sum())}（涉及 {int((ok.n_zero_rounds > 0).sum())} 张图）")
    P(f"读取出错的图 {int(maps.get('err_scan', pd.Series(dtype=str)).notna().sum())}，缺文件的图 {int((maps.missing != '').sum())}")
    if "err_scan" in maps:
        for e, c in maps.err_scan.dropna().value_counts().head(5).items():
            P(f"  错误 x{c}: {e}")
    P(f"缺少 bomb 列的 rounds 表 {int((~ok.has_bomb_cols.fillna(True).astype(bool)).sum())}")

    P("\n## 2. 年份")
    P(ok.groupby("year").agg(赛事=("event_id", "nunique"), 比赛=("match_id", "nunique"), 地图=("path", "size"), 回合=("n_rounds", "sum")).to_string())
    P("\n赛事列表（年份 / 场次 / 地图 / 版本号范围）:")
    for _, e in ev.iterrows():
        tag = "" if pd.notna(e.year_from_name) else "  ← 年份按比赛ID推断"
        P(f"  {e.event_id:>4} {int(e.year) if pd.notna(e.year) else '?'} {e.matches:>4}场 {e.maps:>4}图  patch {e.patch_min}-{e.patch_max}  {e.event_name}{tag}")

    P("\n## 3. 完整性")
    dup = ok.groupby("match_id").dataset.nunique()
    P(f"同一场比赛出现在多个数据集: {int((dup > 1).sum())} 场")
    split = ok.groupby(["match_id", "map_name"]).size()
    P(f"同一场比赛同一张地图有多个目录（分段 demo）: {int((split > 1).sum())} 处")
    if len(rep) and "status" in rep:
        P("转换报告状态:", dict(rep.drop_duplicates(["match_id", "map_folder"])["status"].value_counts()))
    P(f"两队识别成功 {int(ok.teams_ok.fillna(False).sum())}/{len(ok)}；选手数分布 {dict(ok.n_players.value_counts().sort_index())}")
    leg = ok[ok.teams_ok.fillna(False).astype(bool)]
    P(f"比分合法（13-x 或加时规则）{int(leg.legal_score.sum())}/{len(leg)}")
    bad = leg[~leg.legal_score.astype(bool)]
    P("  不合法比分示例:", ", ".join(f"{int(a)}-{int(b)}" for a, b in bad[["score_A", "score_B"]].head(15).values))
    P(f"  重复 round_num 的图 {int((ok.dup_round_num > 0).sum())}，winner 异常的图 {int((ok.bad_winner > 0).sum())}")

    P("\n## 4. 选手")
    if len(mp):
        mp2 = mp[mp.team.isin(["A", "B"])]
        P(f"不同 steamid {mp2.steamid.nunique()}")
        alias = mp2.groupby("steamid").name.nunique()
        P(f"用过多个名字的 steamid {int((alias > 1).sum())}，例如:",
          "; ".join("/".join(sorted(mp2[mp2.steamid == s].name.dropna().unique())[:4]) for s in alias[alias > 1].index[:6]))
        py = mp2.groupby(["steamid", "year"]).path.nunique()
        P("每个「选手-年份」的地图数:", f"中位数 {py.median():.0f}，≥20 张: {int((py >= 20).sum())}，≥50: {int((py >= 50).sum())}，≥100: {int((py >= 100).sum())}，共 {len(py)} 个选手-年份")
        yrs = mp2.groupby("steamid").year.nunique()
        P(f"出现在 2 个及以上年份的选手 {int((yrs >= 2).sum())}，3 个及以上 {int((yrs >= 3).sum())}")
        top = mp2.groupby("steamid").agg(name=("name", lambda s: s.mode().iat[0]), maps=("path", "nunique"), years=("year", lambda s: ",".join(str(int(y)) for y in sorted(s.dropna().unique())))).sort_values("maps", ascending=False)
        P("地图数最多的 20 名选手:", "; ".join(f"{r['name']}({r['maps']},{r['years']})" for _, r in top.head(20).iterrows()))
    P("\n## 5. 战队")
    P(f"kills 表拿到队名的图 {int(ok.teamA_name.notna().sum())}/{len(ok)}，示例:",
      "; ".join(f"{a} vs {b}" for a, b in ok[["teamA_name", "teamB_name"]].dropna().head(8).values))
    if len(mp):
        ty = pd.concat([ok[["year", "teamA_name"]].rename(columns={"teamA_name": "t"}), ok[["year", "teamB_name"]].rename(columns={"teamB_name": "t"})]).dropna()
        tyc = ty.groupby(["t", "year"]).size().sort_values(ascending=False)
        P(f"不同队名 {ty.t.nunique()}；「战队-年份」{len(tyc)} 个，≥30 张图的 {int((tyc >= 30).sum())} 个")
        P("地图最多的战队-年份:", "; ".join(f"{t}-{int(y)}({c})" for (t, y), c in tyc.head(20).items()))
    if len(rep):
        P("dem 文件名示例:", "; ".join(rep.dem_name.dropna().head(5)))
    P("\n## 6. 地图池（每年各地图的数量）")
    P(pd.crosstab(ok.map_name, ok.year).to_string())
    P("\n## 7. tick 数据抽样（每个数据集 6 张图）")
    P(f"tick 间隔中位数 {ok.tick_step.median()}，每 tick 行数中位数 {ok.rows_per_tick.median()}，每图 tick 数中位数 {ok.n_ticks.median():.0f}")
    P(ts.drop(columns="path").describe().loc[["mean", "min", "max"]].round(4).to_string())
    txt = "\n".join(L)
    open(f"{OUT}/summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
