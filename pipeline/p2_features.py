"""P2 统计特征：每名选手每回合的表现，以及「选手-年份」「战队-年份」汇总。

输入：clean/{maps,rounds,round_players}.parquet（P1 产出）+ 原始事件表（kills/damages/shots/grenades/bomb）
输出：clean/
    player_rounds.parquet  每回合每名在场选手一行
    player_years.parquet   每名选手每年一行（含攻防两边拆分）
    team_years.parquet     每个队名每年一行
    p2_report.txt
"""
import paths
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

C = paths.CLEAN
TICKRATE = 64
TRADE_TICKS = 5 * TICKRATE
UTIL_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade"}
PROJECTILES = {"CFlashbangProjectile": "flashes", "CSmokeGrenadeProjectile": "smokes", "CHEGrenadeProjectile": "hes",
               "CMolotovProjectile": "molotovs", "CDecoyProjectile": "decoys"}
# 旧版 demo 里闪光和手雷的飞行实体都叫 CBaseCSGrenadeProjectile，用投掷者手里同时消失的道具来判断类型
HELD = {"CFlashbang": "flashes", "CHEGrenade": "hes", "CSmokeGrenade": "smokes", "CMolotovGrenade": "molotovs",
        "CIncendiaryGrenade": "molotovs", "CDecoyGrenade": "decoys"}
NADE_COLS = ["flashes", "smokes", "hes", "molotovs", "decoys"]
NUM_COLS = ["kills", "deaths", "assists", "flash_assists", "hs", "awp_kills", "damage", "util_damage",
            "opening_kill", "opening_death", "trade_kill", "traded", "survived", "kast", "clutch_vs", "clutch_won",
            "shots", "flashes", "smokes", "hes", "molotovs", "decoys", "plants", "defuses", "wallbang_kills", "smoke_kills"]


def read(path, table, cols, fmap):
    """读事件表并把 steamid 列统一成字符串。

    awpy 写出时，带空值的 steamid 列（如 attacker_steamid、assister_steamid）被存成了 float64，
    精度在源文件里就已丢失。这里用「本图选手真实 steamid → float」的映射对回去（同图 10 人不会撞号）。
    """
    f = f"{path}/{table}.parquet"
    have = set(pq.read_schema(f).names)
    df = pq.read_table(f, columns=[c for c in cols if c in have]).to_pandas()
    for c in df.columns:
        if c.endswith("steamid"):
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(fmap)
            else:
                df[c] = df[c].where(df[c].isna(), df[c].astype(str))
    return df


def part_features(task):
    path, rds, rps = task
    try:
        fmap = {float(s): s for s in rps.steamid.unique()}
        keep = set(rds.part_round_num)
        k = read(path, "kills", ["tick", "round_num", "attacker_steamid", "victim_steamid", "assister_steamid",
                                 "assistedflash", "headshot", "weapon", "penetrated", "thrusmoke"], fmap)
        k = k[k.round_num.isin(keep)].sort_values("tick")
        dm = read(path, "damages", ["round_num", "attacker_steamid", "victim_steamid", "dmg_health_real", "weapon",
                                    "attacker_side", "victim_side"], fmap)
        dm = dm[dm.round_num.isin(keep) & (dm.attacker_side != dm.victim_side) & dm.attacker_steamid.notna()]
        dmg = dm.groupby(["round_num", "attacker_steamid"]).dmg_health_real.sum()
        udmg = dm[dm.weapon.isin(UTIL_WEAPONS)].groupby(["round_num", "attacker_steamid"]).dmg_health_real.sum()
        sh = read(path, "shots", ["round_num", "player_steamid"], fmap)
        shots = sh[sh.round_num.isin(keep)].groupby(["round_num", "player_steamid"]).size()
        gr = read(path, "grenades", ["round_num", "thrower_steamid", "grenade_type", "entity_id", "tick"], fmap)
        gr = gr[gr.round_num.isin(keep)]
        # 实体编号会跨回合回收，所以按（回合, 编号, 类型）分组
        ent = gr.groupby(["round_num", "entity_id", "grenade_type"]).agg(
            thrower=("thrower_steamid", "first"), tick=("tick", "min"), last=("tick", "max")).reset_index()
        ent = ent.rename(columns={"grenade_type": "type"})
        proj = ent[ent.type.str.endswith("Projectile")].copy()
        proj["col"] = proj.type.map(PROJECTILES)
        generic = proj[proj.col.isna()].dropna(subset=["thrower"]).sort_values("tick")
        # 飞行物出现时刻通常比手里道具消失早 1–17 tick：在 ±32 tick 内找同一人同一回合最近的闪光/手雷
        held = ent[ent.type.isin(["CFlashbang", "CHEGrenade"])].dropna(subset=["thrower"])
        if len(generic) and len(held):
            h = held.assign(col=held.type.map(HELD))[["round_num", "thrower", "last", "col"]].rename(columns={"last": "tick"}).sort_values("tick")
            mt = pd.merge_asof(generic.drop(columns="col"), h, on="tick", by=["thrower", "round_num"], direction="nearest", tolerance=32)
            proj = pd.concat([proj[proj.col.notna()], mt], ignore_index=True)
        nades = proj.dropna(subset=["col"]).groupby(["round_num", "thrower", "col"]).size()
        bm = read(path, "bomb", ["round_num", "event", "steamid"], fmap)
        bm["event"] = bm["event"].astype(str).str.lower()
        plants = bm[bm.event.eq("plant")].groupby(["round_num", "steamid"]).size()
        defuses = bm[bm.event.eq("defuse")].groupby(["round_num", "steamid"]).size()

        out = []
        for r in rds.itertuples(index=False):
            prn = int(r.part_round_num)
            pl = rps[rps.real_round == r.real_round]
            team_of = dict(zip(pl.steamid, pl.team))
            st = {sid: defaultdict(float) for sid in team_of}
            alive = {tm: {s for s, t in team_of.items() if t == tm} for tm in (1, 2)}
            first_done = False
            deaths = []                      # (tick, victim, killer)
            for e in k[k.round_num == prn].itertuples(index=False):
                v, a, asst = e.victim_steamid, e.attacker_steamid, e.assister_steamid
                vt, at = team_of.get(v), team_of.get(a)
                if vt is None:
                    continue
                st[v]["deaths"] += 1
                st[v]["death_tick"] = e.tick
                if at is not None and at != vt:
                    s = st[a]
                    s["kills"] += 1
                    s["hs"] += bool(e.headshot)
                    s["awp_kills"] += e.weapon == "awp"
                    s["wallbang_kills"] += bool(e.penetrated)
                    s["smoke_kills"] += bool(e.thrusmoke)
                    if "first_kill_tick" not in s:
                        s["first_kill_tick"] = e.tick
                    if not first_done:
                        s["opening_kill"] = 1
                        st[v]["opening_death"] = 1
                        first_done = True
                    for t0, vic, kil in deaths:      # 补枪：杀掉了刚刚杀死队友的人
                        if kil == v and team_of.get(vic) == at and e.tick - t0 <= TRADE_TICKS:
                            s["trade_kill"] = 1
                            st[vic]["traded"] = 1
                    if asst in st and team_of.get(asst) == at:
                        st[asst]["assists"] += 1
                        st[asst]["flash_assists"] += bool(e.assistedflash)
                deaths.append((e.tick, v, a))
                alive[vt].discard(v)
                for tm in (1, 2):                    # 残局：某队只剩 1 人时记录面对几个敌人
                    if len(alive[tm]) == 1 and alive[3 - tm]:
                        sid = next(iter(alive[tm]))
                        if st[sid]["clutch_vs"] == 0:
                            st[sid]["clutch_vs"] = len(alive[3 - tm])
            for x in pl.itertuples(index=False):
                sid = x.steamid
                s = st[sid]
                s["survived"] = float(s["deaths"] == 0)
                s["kast"] = float(s["kills"] > 0 or s["assists"] > 0 or s["survived"] > 0 or s["traded"] > 0)
                s["clutch_won"] = float(s["clutch_vs"] > 0 and r.winner_team == x.team)
                key = (prn, sid)
                s["damage"] = float(dmg.get(key, 0))
                s["util_damage"] = float(udmg.get(key, 0))
                s["shots"] = float(shots.get(key, 0))
                for col in NADE_COLS:
                    s[col] = float(nades.get((prn, sid, col), 0))
                s["plants"] = float(plants.get(key, 0))
                s["defuses"] = float(defuses.get(key, 0))
                row = {"map_uid": r.map_uid, "real_round": r.real_round, "steamid": sid, "team": x.team, "side": x.side,
                       "equip": x.equip, "won": int(r.winner_team == x.team)}
                row.update({c: s.get(c, 0.0) for c in NUM_COLS})
                row["first_kill_time"] = (s["first_kill_tick"] - r.freeze_end) / TICKRATE if "first_kill_tick" in s else np.nan
                row["death_time"] = (s["death_tick"] - r.freeze_end) / TICKRATE if "death_tick" in s else np.nan
                out.append(row)
        return pd.DataFrame(out), None
    except Exception as e:
        return pd.DataFrame(), f"{path}: {type(e).__name__}: {e}"[:300]


def summarize(df, keys):
    g = df.groupby(keys)
    s = g.agg(maps=("map_uid", "nunique"), rounds=("real_round", "size"),
              **{c: (c, "sum") for c in ["kills", "deaths", "assists", "flash_assists", "hs", "awp_kills", "damage",
                                         "util_damage", "opening_kill", "opening_death", "trade_kill", "traded",
                                         "survived", "kast", "flashes", "smokes", "hes", "molotovs", "plants", "defuses"]},
              clutches=("clutch_vs", lambda x: int((x > 0).sum())), clutch_wins=("clutch_won", "sum"),
              avg_equip=("equip", "mean"), first_kill_time=("first_kill_time", "median")).reset_index()
    n = s["rounds"]
    s["kpr"], s["dpr"], s["apr"], s["adr"] = s.kills / n, s.deaths / n, s.assists / n, s.damage / n
    s["kast_pct"] = s.kast / n
    s["hs_pct"] = s.hs / s.kills.where(s.kills > 0)
    s["awp_kill_share"] = s.awp_kills / s.kills.where(s.kills > 0)
    s["opening_attempts_pr"] = (s.opening_kill + s.opening_death) / n
    s["opening_success"] = s.opening_kill / (s.opening_kill + s.opening_death).where(lambda x: x > 0)
    s["trade_kill_pr"], s["traded_pr"] = s.trade_kill / n, s.traded / n
    s["util_dmg_pr"], s["flashes_pr"], s["flash_assists_pr"] = s.util_damage / n, s.flashes / n, s.flash_assists / n
    s["survival_pct"] = s.survived / n
    s["clutch_win_pct"] = s.clutch_wins / s.clutches.where(s.clutches > 0)
    impact = 2.13 * s.kpr + 0.42 * s.apr - 0.41
    s["rating_approx"] = 0.0073 * s.kast_pct * 100 + 0.3591 * s.kpr - 0.5329 * s.dpr + 0.2372 * impact + 0.0032 * s.adr + 0.1587
    return s


def main(limit=None):
    t0 = time.time()
    maps = pd.read_parquet(f"{C}/maps.parquet")
    rd = pd.read_parquet(f"{C}/rounds.parquet")
    rp = pd.read_parquet(f"{C}/round_players.parquet")
    rp["steamid"] = rp["steamid"].astype(str)
    tasks = []
    rp_by_map = dict(tuple(rp.groupby("map_uid")))
    for path, g in rd.groupby("part_path"):
        uid = g.map_uid.iat[0]
        tasks.append((path, g[["map_uid", "real_round", "part_round_num", "freeze_end", "winner_team"]].copy(),
                      rp_by_map.get(uid, rp.iloc[:0])))
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} 个分段待处理……", flush=True)
    parts, errs = [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for n, (df, err) in enumerate(ex.map(part_features, tasks, chunksize=2), start=1):
            parts.append(df)
            if err:
                errs.append(err)
            if n % 1000 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    pr = pd.concat(parts, ignore_index=True)
    pr = pr.merge(maps[["map_uid", "year", "event_id", "map_name", "team1_slug", "team2_slug"]], on="map_uid", how="left")
    pr["team_slug"] = np.where(pr.team == 1, pr.team1_slug, pr.team2_slug)
    pr = pr.drop(columns=["team1_slug", "team2_slug"])
    pr.to_parquet(f"{C}/player_rounds.parquet", index=False)

    summarize_tables(pr, t0, len(tasks), errs)


def summarize_tables(pr=None, t0=None, n_tasks=0, errs=()):
    """由 player_rounds 重算选手-年份、战队-年份汇总并写报告（全量流程和增量更新共用）。"""
    t0 = t0 or time.time()
    pr = pr if pr is not None else pd.read_parquet(f"{C}/player_rounds.parquet")
    names = pd.read_parquet(f"{C}/players.parquet")[["steamid", "name"]]
    py = summarize(pr, ["steamid", "year"])
    for side in ("ct", "t"):
        sd = summarize(pr[pr.side == side], ["steamid", "year"])[["steamid", "year", "kpr", "adr", "kast_pct", "opening_attempts_pr", "rating_approx"]]
        py = py.merge(sd.rename(columns={c: f"{side}_{c}" for c in sd.columns if c not in ("steamid", "year")}), on=["steamid", "year"], how="left")
    py = names.merge(py, on="steamid", how="right")
    py.to_parquet(f"{C}/player_years.parquet", index=False)
    ty = summarize(pr.dropna(subset=["team_slug"]), ["team_slug", "year"])
    ty["round_win_pct"] = pr.dropna(subset=["team_slug"]).groupby(["team_slug", "year"]).won.mean().values
    ty.to_parquet(f"{C}/team_years.parquet", index=False)

    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("# P2 特征报告", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s")
    P(f"分段 {n_tasks}，出错 {len(errs)}；player_rounds {len(pr)} 行，覆盖 {pr.map_uid.nunique()} 张图")
    for e in errs[:5]:
        P("  错误:", e)
    tot = pr[["kills", "deaths"]].sum()
    P(f"总击杀 {int(tot.kills)}，总死亡 {int(tot.deaths)}（死亡含自杀/误杀，应略多于击杀）")
    P(f"每回合平均：击杀 {pr.groupby(['map_uid', 'real_round']).kills.sum().mean():.2f}，开局击杀 {pr.groupby(['map_uid', 'real_round']).opening_kill.sum().mean():.3f}（应接近 1）")
    P(f"全体 KPR {pr.kills.mean():.3f}，ADR {pr.damage.mean():.1f}，KAST {pr.kast.mean():.1%}，残局胜率 {pr[pr.clutch_vs > 0].clutch_won.mean():.1%}")
    big = py[py.rounds >= 500].sort_values("rating_approx", ascending=False)
    P("\n每年 rating_approx 前 5（至少 500 回合）:")
    for y, g in big.groupby("year"):
        P(f"  {int(y)}: " + "; ".join(f"{r['name']} {r.rating_approx:.2f}" for _, r in g.head(5).iterrows()))
    P("\n狙击手（AWP 击杀占比最高，至少 500 回合）:", "; ".join(f"{r['name']}-{int(r.year)} {r.awp_kill_share:.0%}" for _, r in big.sort_values("awp_kill_share", ascending=False).head(8).iterrows()))
    P("首杀尝试最多（突破手）:", "; ".join(f"{r['name']}-{int(r.year)} {r.opening_attempts_pr:.2f}" for _, r in big.sort_values("opening_attempts_pr", ascending=False).head(8).iterrows()))
    P("闪光最多（辅助）:", "; ".join(f"{r['name']}-{int(r.year)} {r.flashes_pr:.2f}" for _, r in big.sort_values("flashes_pr", ascending=False).head(8).iterrows()))
    tyb = ty[ty.rounds >= 1000].sort_values("round_win_pct", ascending=False)
    P("\n回合胜率最高的战队-年份（至少 1000 回合）:", "; ".join(f"{r.team_slug}-{int(r.year)} {r.round_win_pct:.1%}" for _, r in tyb.head(10).iterrows()))
    txt = "\n".join(L)
    open(f"{C}/p2_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
