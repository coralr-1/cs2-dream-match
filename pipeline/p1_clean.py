"""P1 数据清洗：拼接分段 demo、去重、识别两队、对齐真实回合号、推断缺失回合、推算队名。

输入：$CS2_ROOT/cs2-s-tier-demos-*/（原始数据）+ _index/maps.csv、reports.csv（audit.py 生成）
输出：$CS2_CLEAN/
    maps.parquet           每张图一行（分段已合并）：两队阵容、队名、最终比分、质量标记
    rounds.parquet         每个实际记录到的回合一行：真实回合号、胜方、经济、回合前比分
    round_players.parquet  每回合每名在场选手一行：所属队伍、阵营、冻结结束时的装备/血量/护甲
    players.parquet        每名选手一行：常用名、曾用名、地图数、年份
    p1_report.txt          清洗报告

真实回合号的推断规则（在原始数据上验证过）：
- CS2 MR12：上半场 12 回合，下半场 12 回合；加时每半 3 回合，且加时开始时不换边。
  所以「同一队守 CT」的连续回合段长度依次应为 12、15、6、6……（最后一段可以更短）。
- 手枪局（双方平均装备 ≤ 1200）只出现在第 1 和第 13 回合。
- 旧版 demo 常常漏掉开局手枪局；分段 demo 的衔接处可能漏掉或重复一回合；中途重开会在段中间出现手枪局。
- 缺失回合的胜方用「最终比分必须合法」来确定，必要时参考下一回合的经济。
"""
import paths
import itertools
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

B = paths.ROOT
IDX = os.environ.get("CS2_INDEX", f"{B}/_index")
OUT = paths.CLEAN
PISTOL_MAX = 1200
MAPS_RE = "ancient|anubis|cache|dust2|dust|inferno|mirage|nuke|overpass|train|vertigo"
TICK_COLS = ["tick", "steamid", "name", "side", "health", "armor", "has_helmet", "has_defuser", "current_equip_value"]


# ---------------------------------------------------------------- 工具函数
def parse_dem(name):
    """'natus-vincere-vs-faze-m3-inferno-p2.dem' -> ('natus-vincere', 'faze', 2)"""
    if not isinstance(name, str):
        return None, None, None
    s = re.sub(r"\.dem$", "", name)
    part = None
    m = re.search(r"-(?:p|з)(\d+)$", s) or re.search(r"-(\d)$", s)
    if m:
        part, s = int(m.group(1)), s[:m.start()]
    if "-vs-" not in s:
        return None, None, part
    a, rest = s.split("-vs-", 1)
    rest = re.sub(rf"-(?:m\d-)?(?:{MAPS_RE})$", "", rest)
    rest = re.sub(r"-m\d-.*$", "", rest)
    return a, rest, part


def legal_score(a, b):
    hi, lo = max(a, b), min(a, b)
    if hi == 13 and lo <= 11:
        return True
    if hi < 16 or (hi - 13) % 3:
        return False
    return hi - 4 <= lo <= hi - 2


def expected_len(k):
    """第 k 个「同一队守 CT」段的应有长度（不是最后一段时）。"""
    return 12 if k == 0 else (15 if k == 1 else 6)


# ---------------------------------------------------------------- 读取一个分段
def load_part(path):
    rd = pd.read_parquet(f"{path}/rounds.parquet")
    rd = rd.dropna(subset=["start", "end"])
    rd = rd[rd["start"] != rd["end"]].sort_values("start").reset_index(drop=True)
    # 录像从回合中途开始时 freeze_end 为空（PGL 等赛事的开场假回合就是这样）：用回合开始代替，
    # 是否为假回合交给后面的「换边节奏 + 手枪局」逻辑判断
    rd["freeze_end"] = rd["freeze_end"].fillna(rd["start"])
    tk = pq.read_table(f"{path}/ticks.parquet", columns=TICK_COLS).to_pandas()
    ticks = np.sort(tk["tick"].unique())
    idx = np.clip(np.searchsorted(ticks, rd["freeze_end"].values), 0, len(ticks) - 1)
    rd["fe_tick"] = ticks[idx]
    snap = tk[tk["tick"].isin(set(rd["fe_tick"]))]
    snap = snap[(snap["health"] > 0) & snap["side"].isin(["ct", "t"])]
    return rd, snap


# ---------------------------------------------------------------- 处理一张图（可能由多个分段组成）
def build_map(task):
    map_uid, parts = task
    flags = []
    recs, snaps = [], []
    for pi, p in enumerate(parts):
        rd, snap = load_part(p["path"])
        for _, r in rd.iterrows():
            s = snap[snap["tick"] == r["fe_tick"]]
            ct, t = s[s["side"] == "ct"], s[s["side"] == "t"]
            recs.append({
                "part": pi, "part_path": p["path"], "part_round_num": int(r["round_num"]),
                "start": int(r["start"]), "freeze_end": int(r["freeze_end"]), "end": int(r["end"]),
                "winner_side": str(r["winner"]).lower(), "reason": str(r.get("reason")),
                "bomb_plant": r.get("bomb_plant"), "bomb_site": r.get("bomb_site"),
                "ct_set": frozenset(ct["steamid"]), "t_set": frozenset(t["steamid"]),
                "ct_n": len(ct), "t_n": len(t),
                "ct_eq": float(ct["current_equip_value"].mean()) if len(ct) else np.nan,
                "t_eq": float(t["current_equip_value"].mean()) if len(t) else np.nan,
                "orig": len(recs),
            })
            snaps.append(s.assign(rec=len(recs) - 1))
    if not recs:
        return {"map_uid": map_uid, "flags": "no_rounds"}, [], []
    snap_all = pd.concat(snaps, ignore_index=True)

    # ---- 识别两队：第一个 5v5 回合定义 team1(CT)/team2(T)，其余回合按重合人数归队
    team1 = team2 = None
    for r in recs:
        if len(r["ct_set"]) == 5 and len(r["t_set"]) == 5:
            team1, team2 = set(r["ct_set"]), set(r["t_set"])
            break
    if team1 is None:
        r = max(recs, key=lambda x: min(len(x["ct_set"]), len(x["t_set"])))
        team1, team2 = set(r["ct_set"]), set(r["t_set"])
        flags.append("no_5v5")
    for r in recs:
        a1 = len(r["ct_set"] & team1) + len(r["t_set"] & team2)
        a2 = len(r["ct_set"] & team2) + len(r["t_set"] & team1)
        r["ct_team"] = 1 if a1 > a2 else (2 if a2 > a1 else None)
    last = None
    for r in recs:                      # 无法判断的回合沿用上一回合
        if r["ct_team"] is None:
            r["ct_team"] = last
        last = r["ct_team"]
    first = next((r["ct_team"] for r in recs if r["ct_team"]), 1)
    for r in recs:
        if r["ct_team"] is None:
            r["ct_team"] = first
        r["pistol"] = bool(np.nanmax([r["ct_eq"], r["t_eq"]]) <= PISTOL_MAX) if r["ct_n"] and r["t_n"] else False

    # ---- 按「同一队守 CT」切段
    def make_runs(rs):
        runs = []
        for i, r in enumerate(rs):
            if runs and rs[runs[-1][-1]]["ct_team"] == r["ct_team"]:
                runs[-1].append(i)
            else:
                runs.append([i])
        return runs

    # ---- 处理中途重开：段内出现不该有的手枪局，丢弃它之前的回合。
    # 为避免把「双方同时存钱」误判为手枪局，只在段长超标或手枪局前的回合来自更早的分段时才判定为重开。
    n_dropped = 0
    for _ in range(5):
        runs = make_runs(recs)
        cut = None
        for k, run in enumerate(runs):
            pist = [j for j, i in enumerate(run) if recs[i]["pistol"]]
            if k <= 1:
                for j in sorted((j for j in pist if j > 0), reverse=True):
                    other_part = recs[run[j - 1]]["part"] != recs[run[j]]["part"]
                    if len(run) - j <= expected_len(k) and (len(run) > expected_len(k) or other_part):
                        cut = (run[0], run[j])                          # 丢弃 [run[0], run[j])
                        break
            elif pist and pist[0] == 0:                                 # 加时段以手枪局开头 → 整场重开
                cut = (0, run[0])
            if cut:
                break
        if cut is None:
            break
        lo, hi = cut
        n_dropped += hi - lo
        recs = recs[:lo] + recs[hi:]
        flags.append("restart")
    runs = make_runs(recs)

    # ---- 处理过长的段：分段衔接处重复的回合，丢掉前一段末尾的
    keep = [True] * len(recs)
    for k, run in enumerate(runs):
        E = expected_len(k)
        if len(run) > E:
            extra = len(run) - E
            bounds = [j for j in range(1, len(run)) if recs[run[j]]["part"] != recs[run[j - 1]]["part"]]
            if bounds:
                b = bounds[0]
                for j in range(max(0, b - extra), b):
                    keep[run[j]] = False
                flags.append("overlap_dropped")
            else:
                flags.append("overlong_run")
    if not all(keep):
        n_dropped += keep.count(False)
        recs = [r for r, kp in zip(recs, keep) if kp]
        runs = make_runs(recs)

    # ---- 排出真实回合序列，缺失回合用占位符
    seq = []          # ("rec", idx) 或 ("miss", is_pistol_slot)
    for k, run in enumerate(runs):
        E = expected_len(k)
        is_last = k == len(runs) - 1
        L = len(run)
        miss = 0 if is_last else max(E - L, 0)
        start_miss = 0
        if k <= 1 and not recs[run[0]]["pistol"]:
            start_miss = 1 if (is_last or miss >= 1) else 0
            if is_last:
                miss = 1
        rest = miss - start_miss
        bounds = [j for j in range(1, L) if recs[run[j]]["part"] != recs[run[j - 1]]["part"]]
        if k == 0 and rest > 0 and not bounds:
            start_miss += rest                      # 第一段又没有衔接点：视为 demo 开录晚了
            rest = 0
        for s in range(start_miss):
            seq.append(("miss", k <= 1 and s == 0))
        for j, i in enumerate(run):
            if rest and bounds and j == bounds[0]:
                seq.extend([("miss", False)] * rest)
                rest = 0
            seq.append(("rec", i))
        seq.extend([("miss", False)] * rest)

    def win_team(r):
        return r["ct_team"] if r["winner_side"] == "ct" else (3 - r["ct_team"])

    base = Counter(win_team(recs[i]) for kind, i in seq if kind == "rec" and recs[i]["winner_side"] in ("ct", "t"))
    miss_pos = [p for p, (kind, _) in enumerate(seq) if kind == "miss"]
    # 缺失的手枪局：猜下一回合装备更好的一方赢了手枪局
    guess = {}
    for p in miss_pos:
        if seq[p][1]:
            nxt = next((recs[i] for kind, i in seq[p + 1:] if kind == "rec"), None)
            if nxt is not None and not np.isnan(nxt["ct_eq"]) and not np.isnan(nxt["t_eq"]):
                guess[p] = nxt["ct_team"] if nxt["ct_eq"] > nxt["t_eq"] else 3 - nxt["ct_team"]

    cands = []
    combos = itertools.product([1, 2], repeat=len(miss_pos)) if len(miss_pos) <= 10 else [tuple(guess.get(p, 1) for p in miss_pos)]
    for combo in combos:
        s1 = base[1] + combo.count(1)
        s2 = base[2] + combo.count(2)
        hits = sum(1 for p, w in zip(miss_pos, combo) if guess.get(p) == w)
        for extra in (0, 1):
            f1, f2 = s1, s2
            if extra:
                if f1 == f2:
                    continue
                f1, f2 = (f1 + 1, f2) if f1 > f2 else (f1, f2 + 1)
            if legal_score(f1, f2):
                cands.append((extra, -hits, f1, f2, combo))
    if cands:
        cands.sort(key=lambda c: (c[0], c[1]))
        best = cands[0]
        extra, _, f1, f2, combo = best
        ambiguous = len({(c[2], c[3]) for c in cands if c[0] == best[0] and c[1] == best[1]}) > 1
        legal = True
    else:
        extra, f1, f2, combo, ambiguous, legal = 0, base[1], base[2], tuple(guess.get(p, 0) for p in miss_pos), False, False
        flags.append("illegal_score")
    miss_winner = dict(zip(miss_pos, combo))

    # ---- 输出回合表
    rounds, rp = [], []
    s1 = s2 = 0
    for real, (kind, i) in enumerate(seq, start=1):
        if kind == "miss":
            w = miss_winner.get(real - 1)
        else:
            r = recs[i]
            w = win_team(r) if r["winner_side"] in ("ct", "t") else None
            half = 1 if real <= 12 else (2 if real <= 24 else 3)
            rounds.append({
                "map_uid": map_uid, "real_round": real, "half": half,
                "ot_index": 0 if real <= 24 else (real - 25) // 6 + 1,
                "part": r["part"], "part_path": r["part_path"], "part_round_num": r["part_round_num"],
                "start": r["start"], "freeze_end": r["freeze_end"], "end": r["end"],
                "winner_side": r["winner_side"], "winner_team": w, "ct_team": r["ct_team"], "reason": r["reason"],
                "bomb_plant": r["bomb_plant"], "bomb_site": r["bomb_site"],
                "ct_n": r["ct_n"], "t_n": r["t_n"], "ct_eq": r["ct_eq"], "t_eq": r["t_eq"], "pistol": r["pistol"],
                "score1_before": s1, "score2_before": s2,
            })
            s = snap_all[snap_all["rec"] == r["orig"]]
            for _, x in s.iterrows():
                side = x["side"]
                team = r["ct_team"] if side == "ct" else 3 - r["ct_team"]
                rp.append({"map_uid": map_uid, "real_round": real, "steamid": str(x["steamid"]), "name": x["name"],
                           "team": team, "side": side, "equip": int(x["current_equip_value"]), "health": int(x["health"]),
                           "armor": int(x["armor"]), "helmet": bool(x["has_helmet"]), "defuser": bool(x["has_defuser"])})
        if w == 1:
            s1 += 1
        elif w == 2:
            s2 += 1

    # ---- 两队阵容（按出场回合数取前 5 名为主力）
    rpd = pd.DataFrame(rp)
    roster = {1: [], 2: []}
    if len(rpd):
        cnt = rpd.groupby(["team", "steamid"]).size().reset_index(name="n").sort_values("n", ascending=False)
        for tm in (1, 2):
            roster[tm] = cnt[cnt.team == tm].steamid.head(5).tolist()
    info = {
        "map_uid": map_uid, "n_parts": len(parts), "parts": "|".join(p["path"] for p in parts),
        "n_rounds_recorded": len(rounds), "n_rounds_real": len(seq) + extra,
        "n_missing": len(miss_pos), "final_round_missing": bool(extra), "n_dropped": n_dropped,
        "score1": f1, "score2": f2, "winner_team": 1 if f1 > f2 else (2 if f2 > f1 else 0),
        "legal": legal, "ambiguous": ambiguous, "n_runs": len(runs),
        "roster1": ",".join(roster[1]), "roster2": ",".join(roster[2]),
        "flags": ",".join(sorted(set(flags))),
    }
    return info, rounds, rp


def safe_build(task):
    try:
        return build_map(task)
    except Exception as e:
        return {"map_uid": task[0], "flags": "error", "error": f"{type(e).__name__}: {e}"[:200]}, [], []


# ---------------------------------------------------------------- 主流程
def main(limit=None):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    maps = pd.read_csv(f"{IDX}/maps.csv", dtype={"match_id": str, "patch": str})
    rep = pd.read_csv(f"{IDX}/reports.csv", dtype=str).drop_duplicates(["match_id", "map_folder"])

    # 去掉跨数据集的完全重复（同一比赛、同一目录名、同样 tick 行数）
    n0 = len(maps)
    maps = maps.sort_values("dataset").drop_duplicates(["match_id", "map_folder", "rows_ticks"])
    n_dup = n0 - len(maps)
    maps = maps.merge(rep[["match_id", "map_folder", "dem_name"]], on=["match_id", "map_folder"], how="left")
    parsed = maps["dem_name"].map(parse_dem)
    maps["slug_a"] = parsed.map(lambda x: x[0])
    maps["slug_b"] = parsed.map(lambda x: x[1])
    maps["dem_part"] = parsed.map(lambda x: x[2])
    maps["folder_idx"] = maps["map_folder"].str.extract(r"^m(\d+)_")[0].astype(float)
    maps["map_uid"] = maps["match_id"] + "_" + maps["map_name"]

    tasks = []
    for uid, g in maps.groupby("map_uid"):
        g = g.sort_values(["dem_part", "folder_idx"], na_position="first")
        tasks.append((uid, [{"path": p} for p in g["path"]]))
    if limit:
        tasks = tasks[:limit]
    print(f"去重 {n_dup} 个重复目录；共 {len(tasks)} 张图，开始处理……", flush=True)

    infos, rounds, rps = [], [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for n, (info, r, p) in enumerate(ex.map(safe_build, tasks, chunksize=2), start=1):
            infos.append(info); rounds.extend(r); rps.extend(p)
            if n % 500 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    mi = pd.DataFrame(infos)
    rd = pd.DataFrame(rounds)
    rp = pd.DataFrame(rps)

    # ---- 比赛级信息
    first = maps.sort_values(["dem_part", "folder_idx"], na_position="first").groupby("map_uid").first()
    meta = first[["match_id", "map_name", "event_id", "event_name", "year", "patch", "dataset"]].reset_index()
    order = maps.groupby(["match_id", "map_name"]).folder_idx.min().reset_index()
    order["map_number"] = order.groupby("match_id").folder_idx.rank(method="dense").astype(int)
    meta = meta.merge(order[["match_id", "map_name", "map_number"]], on=["match_id", "map_name"], how="left")
    # 每场比赛的队名对：取该比赛各图里最常见的解析结果
    pair = maps.dropna(subset=["slug_a", "slug_b"]).groupby("match_id").apply(
        lambda g: Counter(zip(g.slug_a, g.slug_b)).most_common(1)[0][0]).rename("pair")
    meta = meta.merge(pair, left_on="match_id", right_index=True, how="left")
    mi = meta.merge(mi, on="map_uid", how="right")

    # ---- 队名分配：选手和队名在同一赛事/同一年/全部数据里的共现次数，决定阵容对应哪个队名
    cnt = {lvl: defaultdict(Counter) for lvl in ("event", "year", "all")}
    for _, m in mi.dropna(subset=["pair"]).iterrows():
        for ro in (m.roster1, m.roster2):
            for sid in str(ro).split(","):
                for slug in m.pair:
                    cnt["event"][(m.event_id, sid)][slug] += 1
                    cnt["year"][(m.year, sid)][slug] += 1
                    cnt["all"][sid][slug] += 1

    def assign(m):
        if not isinstance(m.pair, tuple):
            return pd.Series({"team1_slug": None, "team2_slug": None, "slug_level": "none"})
        a, b = m.pair
        r1, r2 = str(m.roster1).split(","), str(m.roster2).split(",")
        for lvl, key in (("event", lambda s: (m.event_id, s)), ("year", lambda s: (m.year, s)), ("all", lambda s: s)):
            sc = lambda ro, slug: sum(cnt[lvl][key(s)][slug] for s in ro)
            straight = sc(r1, a) + sc(r2, b)
            swapped = sc(r1, b) + sc(r2, a)
            if straight != swapped:
                return pd.Series({"team1_slug": a if straight > swapped else b,
                                  "team2_slug": b if straight > swapped else a, "slug_level": lvl})
        return pd.Series({"team1_slug": a, "team2_slug": b, "slug_level": "tie"})

    mi = pd.concat([mi, mi.apply(assign, axis=1)], axis=1).drop(columns=["pair"])

    # ---- 选手表
    if len(rp):
        rp = rp.merge(mi[["map_uid", "year"]], on="map_uid", how="left")
        pl = rp.groupby("steamid").agg(
            name=("name", lambda s: s.mode().iat[0]),
            aliases=("name", lambda s: "|".join(sorted(set(map(str, s))))),
            n_maps=("map_uid", "nunique"), n_rounds=("real_round", "size"),
            years=("year", lambda s: ",".join(str(int(y)) for y in sorted(s.dropna().unique())))).reset_index()
        rp = rp.drop(columns=["year"])
    else:
        pl = pd.DataFrame()

    # ---- 写出
    for c in ("bomb_plant",):
        if c in rd:
            rd[c] = pd.to_numeric(rd[c], errors="coerce")
    if "bomb_site" in rd:
        rd["bomb_site"] = rd["bomb_site"].astype(str)
    mi.to_parquet(f"{OUT}/maps.parquet", index=False)
    rd.to_parquet(f"{OUT}/rounds.parquet", index=False)
    rp.to_parquet(f"{OUT}/round_players.parquet", index=False)
    pl.to_parquet(f"{OUT}/players.parquet", index=False)

    # ---- 报告
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    ok = mi[mi["flags"].fillna("") != "error"]
    P("# P1 清洗报告", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s")
    P(f"去掉重复目录 {n_dup}；合并后地图 {len(mi)}（其中分段合并 {int((mi.n_parts > 1).sum())}）；出错 {int((mi['flags'] == 'error').sum())}")
    if "error" in mi:
        for e, c in mi.error.dropna().value_counts().head(5).items():
            P(f"  错误 x{c}: {e}")
    P(f"比分合法 {int(ok.legal.sum())}/{len(ok)} = {ok.legal.mean():.2%}；比分有歧义 {int(ok.ambiguous.sum())}")
    P(f"记录到的回合 {len(rd)}，推断的真实回合 {int(ok.n_rounds_real.sum())}；有缺失回合的图 {int((ok.n_missing > 0).sum())}，补最后一回合 {int(ok.final_round_missing.sum())}")
    P(f"丢弃的回合（重开/重复）{int(ok.n_dropped.sum())}，涉及 {int((ok.n_dropped > 0).sum())} 张图")
    P("缺失回合数分布:", ok.n_missing.value_counts().sort_index().to_dict())
    P("标记统计:", Counter(f for fs in ok["flags"].fillna("") for f in fs.split(",") if f))
    P("最终比分最常见:", Counter(f"{max(a, b)}-{min(a, b)}" for a, b in ok[["score1", "score2"]].values).most_common(12))
    P("不合法比分示例:", ", ".join(f"{int(a)}-{int(b)}" for a, b in ok[~ok.legal.astype(bool)][["score1", "score2"]].head(20).values))
    P(f"队名分配：{dict(ok.slug_level.value_counts())}")
    P(f"选手 {len(pl)}；回合-选手记录 {len(rp)}；每回合在场人数分布:", rp.groupby(["map_uid", "real_round"]).size().value_counts().head(6).to_dict() if len(rp) else {})
    P("\n抽查:")
    for uid in ("2367125_de_nuke", "2367122_de_mirage", "2367257_de_mirage", "2367124_de_mirage", "2391011_de_inferno"):
        x = ok[ok.map_uid == uid]
        if len(x):
            x = x.iloc[0]
            P(f"  {uid}: {x.team1_slug} {x.score1}-{x.score2} {x.team2_slug} | 真实回合 {x.n_rounds_real} 记录 {x.n_rounds_recorded} 缺失 {x.n_missing} 丢弃 {x.n_dropped} 合法 {x.legal} 标记 {x['flags']}")
    txt = "\n".join(L)
    open(f"{OUT}/p1_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
