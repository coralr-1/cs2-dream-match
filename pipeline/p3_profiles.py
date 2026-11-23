"""P3 习惯画像：站位快照、选手常站点位与角色、战队战术习惯。

输入：clean/{maps,rounds,round_players,player_rounds,player_years}.parquet + 原始 ticks
输出：clean/
    positions.parquet       每回合冻结结束后若干时刻，每名选手的坐标/点位/武器/是否存活
    place_centers.parquet   每张地图每个点位名称的坐标中心（画热力图时标注用）
    player_places.parquet   选手-年份-地图-攻防-时刻 → 常站点位及占比
    player_roles.parquet    选手-年份 → 角色标签与关键指标
    team_habits.parquet     战队-年份-地图 → 下包点分布、下包时间、CT 常用站位、手枪局/经济局胜率
    p3_report.txt
"""
import paths
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

C = paths.CLEAN
TICKRATE = 64
OFFSETS = [5, 10, 15, 20, 30, 40, 55, 70]          # 冻结结束后的秒数
MIN_ROUNDS_ROLE = 300
AWP_SHARE = 0.25                                  # AWP 击杀占比呈双峰：步枪手 < 5%，狙击手 > 30%


def part_positions(task):
    path, rds = task
    try:
        targets = {}
        for r in rds.itertuples(index=False):
            for t in OFFSETS:
                tick = int(r.freeze_end + t * TICKRATE)
                if tick <= r.end:
                    targets[tick] = (r.map_uid, r.real_round, t)
        if not targets:
            return pd.DataFrame(), pd.DataFrame(), None
        tbl = pq.read_table(f"{path}/ticks.parquet", columns=["tick", "steamid", "side", "health", "X", "Y", "Z", "place", "active_weapon"])
        tbl = tbl.filter(pc.is_in(tbl["tick"], value_set=__import__("pyarrow").array(list(targets), type=tbl.schema.field("tick").type)))
        df = tbl.to_pandas()
        df = df[df.side.isin(["ct", "t"])]
        meta = pd.DataFrame([(k, *v) for k, v in targets.items()], columns=["tick", "map_uid", "real_round", "t"])
        df = df.merge(meta, on="tick")
        df["steamid"] = df["steamid"].astype(str)
        df["alive"] = df.health > 0
        # rounds 表的 bomb_site 有 bug（几乎全是 B），下包点以 bomb 表的 plant 事件为准
        b = pd.read_parquet(f"{path}/bomb.parquet", columns=["tick", "event", "bombsite", "round_num"])
        b = b[b.event == "plant"].drop_duplicates("round_num", keep="last")
        rmap = rds.set_index("part_round_num")
        b = b[b.round_num.isin(rmap.index)]
        plants = pd.DataFrame({"map_uid": rmap.loc[b.round_num, "map_uid"].values, "real_round": rmap.loc[b.round_num, "real_round"].values,
                               "site": b.bombsite.astype(str).str.replace("Bombsite", "").values, "plant_tick": b.tick.values})
        return df.drop(columns=["tick", "health"]), plants, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"{path}: {type(e).__name__}: {e}"[:300]


def top_places(df, keys, k=3):
    c = df.groupby(keys + ["place"]).size().rename("n").reset_index()
    tot = c.groupby(keys).n.transform("sum")
    c["share"] = c.n / tot
    c["rounds"] = tot
    c = c.sort_values(keys + ["share"], ascending=[True] * len(keys) + [False])
    return c.groupby(keys).head(k)


def main(limit=None):
    t0 = time.time()
    maps = pd.read_parquet(f"{C}/maps.parquet")
    rd = pd.read_parquet(f"{C}/rounds.parquet")
    tasks = [(p, g[["map_uid", "real_round", "part_round_num", "freeze_end", "end"]]) for p, g in rd.groupby("part_path")]
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} 个分段，提取站位快照……", flush=True)
    parts, plants, errs = [], [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for n, (df, pl_, err) in enumerate(ex.map(part_positions, tasks, chunksize=2), start=1):
            parts.append(df)
            plants.append(pl_)
            if err:
                errs.append(err)
            if n % 1000 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    pos = pd.concat(parts, ignore_index=True)
    rp = pd.read_parquet(f"{C}/round_players.parquet", columns=["map_uid", "real_round", "steamid", "team"])
    pos = pos.merge(rp, on=["map_uid", "real_round", "steamid"], how="inner")      # 只保留冻结结束时在场的选手（去掉教练）
    pos = pos.merge(maps[["map_uid", "map_name", "year", "team1_slug", "team2_slug"]], on="map_uid", how="left")
    pos["team_slug"] = np.where(pos.team == 1, pos.team1_slug, pos.team2_slug)
    pos = pos.drop(columns=["team1_slug", "team2_slug"])
    pos.to_parquet(f"{C}/positions.parquet", index=False)
    bombs = pd.concat(plants, ignore_index=True)
    bombs.to_parquet(f"{C}/round_bomb.parquet", index=False)

    summarize(maps, rd, pos, bombs, t0, len(tasks), errs)


def summarize(maps=None, rd=None, pos=None, bombs=None, t0=None, n_tasks=0, errs=()):
    """由站位快照、下包表、选手年度表重算习惯画像（全量流程和增量更新共用）。"""
    t0 = t0 or time.time()
    maps = maps if maps is not None else pd.read_parquet(f"{C}/maps.parquet")
    rd = rd if rd is not None else pd.read_parquet(f"{C}/rounds.parquet")
    pos = pos if pos is not None else pd.read_parquet(f"{C}/positions.parquet")
    bombs = bombs if bombs is not None else pd.read_parquet(f"{C}/round_bomb.parquet")
    alive = pos[pos.alive & pos.place.notna() & (pos.place != "")]
    centers = alive.groupby(["map_name", "place"]).agg(X=("X", "median"), Y=("Y", "median"), n=("X", "size")).reset_index()
    centers[centers.n >= 200].to_parquet(f"{C}/place_centers.parquet", index=False)

    # ---- 选手常站点位
    pp = top_places(alive[alive.t.isin([15, 30, 55])], ["steamid", "year", "map_name", "side", "t"])
    pp.to_parquet(f"{C}/player_places.parquet", index=False)

    # ---- 角色
    py = pd.read_parquet(f"{C}/player_years.parquet")
    t30 = alive[(alive.t == 30) & (alive.side == "t")]
    cen = t30.groupby(["map_uid", "real_round", "team"])[["X", "Y"]].transform("sum")
    cnt = t30.groupby(["map_uid", "real_round", "team"]).X.transform("size")
    t30 = t30.assign(dist=np.where(cnt > 1, np.hypot((cen.X - t30.X) / (cnt - 1).clip(lower=1) - t30.X,
                                                    (cen.Y - t30.Y) / (cnt - 1).clip(lower=1) - t30.Y), np.nan))
    lurk = t30.groupby(["steamid", "year"]).dist.mean().rename("lurk_dist").reset_index()
    ct30 = alive[(alive.t == 30) & (alive.side == "ct")]
    anchor = top_places(ct30, ["steamid", "year", "map_name"], k=1).groupby(["steamid", "year"]).share.mean().rename("ct_spot_consistency").reset_index()
    ro = py.merge(lurk, on=["steamid", "year"], how="left").merge(anchor, on=["steamid", "year"], how="left")
    ro = ro[ro.rounds >= MIN_ROUNDS_ROLE].copy()
    rifle = ro[ro.awp_kill_share < AWP_SHARE]
    q_open, q_lurk, q_flash = rifle.t_opening_attempts_pr.quantile(0.75), rifle.lurk_dist.quantile(0.75), rifle.flashes_pr.quantile(0.75)

    def role(r):
        if r.awp_kill_share >= AWP_SHARE:
            return "狙击手"
        if r.t_opening_attempts_pr >= q_open:
            return "突破手"
        if r.lurk_dist >= q_lurk:
            return "自由人"
        if r.flashes_pr >= q_flash:
            return "辅助"
        return "步枪手"
    ro["role"] = ro.apply(role, axis=1)
    keep = ["steamid", "name", "year", "rounds", "maps", "role", "rating_approx", "kpr", "dpr", "adr", "kast_pct", "hs_pct",
            "awp_kill_share", "t_opening_attempts_pr", "opening_success", "lurk_dist", "ct_spot_consistency", "flashes_pr",
            "flash_assists_pr", "util_dmg_pr", "trade_kill_pr", "traded_pr", "survival_pct", "clutch_win_pct", "first_kill_time"]
    ro[keep].to_parquet(f"{C}/player_roles.parquet", index=False)

    # ---- 战队习惯
    rdm = rd.merge(maps[["map_uid", "map_name", "year", "team1_slug", "team2_slug", "legal"]], on="map_uid")
    rdm["t_team"] = 3 - rdm.ct_team
    rdm["t_slug"] = np.where(rdm.t_team == 1, rdm.team1_slug, rdm.team2_slug)
    rdm["ct_slug"] = np.where(rdm.ct_team == 1, rdm.team1_slug, rdm.team2_slug)
    rdm = rdm.merge(bombs, on=["map_uid", "real_round"], how="left")
    rdm["planted"] = rdm.site.notna()
    rdm["plant_time"] = (rdm.plant_tick - rdm.freeze_end) / TICKRATE
    tside = rdm.groupby(["t_slug", "year", "map_name"]).agg(
        t_rounds=("real_round", "size"), t_win_pct=("winner_team", lambda s: np.nan), plant_pct=("planted", "mean"),
        site_a_pct=("site", lambda s: (s == "A").mean()), site_b_pct=("site", lambda s: (s == "B").mean()),
        plant_time_median=("plant_time", "median")).reset_index().rename(columns={"t_slug": "team_slug"})
    tw = rdm.assign(w=rdm.winner_team == rdm.t_team).groupby(["t_slug", "year", "map_name"]).w.mean()
    tside["t_win_pct"] = tw.values
    cw = rdm.assign(w=rdm.winner_team == rdm.ct_team).groupby(["ct_slug", "year", "map_name"]).w.mean().rename("ct_win_pct").reset_index().rename(columns={"ct_slug": "team_slug"})
    # 手枪局、经济局（己方平均装备 < 2000 且对方 > 3500）胜率
    prr = pd.read_parquet(f"{C}/player_rounds.parquet", columns=["map_uid", "real_round", "team", "team_slug", "equip", "won", "map_name", "year"])
    tr_ = prr.groupby(["map_uid", "real_round", "team", "team_slug", "map_name", "year"]).agg(team_eq=("equip", "mean"), won=("won", "max")).reset_index()
    opp = tr_[["map_uid", "real_round", "team", "team_eq"]].assign(team=lambda d: 3 - d.team).rename(columns={"team_eq": "opp_eq"})
    tr_ = tr_.merge(opp, on=["map_uid", "real_round", "team"])
    tr_["pistol"] = tr_.real_round.isin([1, 13])
    tr_["eco"] = (tr_.team_eq < 2000) & (tr_.opp_eq > 3500) & ~tr_.pistol
    tr_["force"] = tr_.team_eq.between(2000, 3800) & (tr_.opp_eq > 4200) & ~tr_.pistol
    econ = tr_.groupby(["team_slug", "year", "map_name"]).apply(lambda g: pd.Series({
        "pistol_win_pct": g[g.pistol].won.mean(), "eco_win_pct": g[g.eco].won.mean(), "force_win_pct": g[g.force].won.mean(),
        "force_rate": g.force.mean()})).reset_index()
    # CT 开局站位组合（第 15 秒五人点位，排序后作为一个组合）
    ct15 = alive[(alive.t == 15) & (alive.side == "ct")]
    setups = ct15.groupby(["map_uid", "real_round", "team_slug", "year", "map_name"]).place.apply(lambda s: " / ".join(sorted(s)) if len(s) == 5 else None).dropna().reset_index()
    top_setup = setups.groupby(["team_slug", "year", "map_name"]).place.agg(
        ct_top_setup=lambda s: Counter(s).most_common(1)[0][0], ct_top_setup_share=lambda s: Counter(s).most_common(1)[0][1] / len(s)).reset_index()
    th = tside.merge(cw, on=["team_slug", "year", "map_name"], how="outer").merge(econ, on=["team_slug", "year", "map_name"], how="left") \
              .merge(top_setup, on=["team_slug", "year", "map_name"], how="left")
    th.to_parquet(f"{C}/team_habits.parquet", index=False)

    # ---- 报告
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("# P3 习惯画像报告", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s")
    P(f"分段 {n_tasks}，出错 {len(errs)}；站位快照 {len(pos)} 行（存活 {int(pos.alive.sum())}）；下包 {len(bombs)} 次，A/B 分布 {bombs.site.value_counts().to_dict()}")
    for e in errs[:5]:
        P("  错误:", e)
    P("各时刻存活率:", pos.groupby("t").alive.mean().round(3).to_dict())
    P("角色分布:", ro.role.value_counts().to_dict())
    for y in sorted(ro.year.unique()):
        g = ro[ro.year == y]
        P(f"  {int(y)} 狙击手: " + ", ".join(g[g.role == '狙击手'].sort_values('rating_approx', ascending=False).name.head(6)))
        P(f"  {int(y)} 突破手: " + ", ".join(g[g.role == '突破手'].sort_values('rating_approx', ascending=False).name.head(6)))
        P(f"  {int(y)} 自由人: " + ", ".join(g[g.role == '自由人'].sort_values('rating_approx', ascending=False).name.head(6)))
    names = dict(zip(py.steamid, py.name))
    for nm, mp, side in [("ZywOo", "de_inferno", "ct"), ("donk", "de_mirage", "t"), ("m0NESY", "de_dust2", "ct")]:
        sids = [s for s, n in names.items() if n == nm]
        x = pp[pp.steamid.isin(sids) & (pp.map_name == mp) & (pp.side == side) & (pp.t == 30)].sort_values(["year", "share"], ascending=[True, False])
        P(f"\n{nm} {mp} {side.upper()} 第30秒常站点位:")
        for y, g in x.groupby("year"):
            P(f"  {int(y)}: " + "; ".join(f"{r.place} {r.share:.0%}" for r in g.itertuples()) + f"（{int(g.rounds.iat[0])} 回合）")
    v = th[(th.team_slug == "vitality") & (th.t_rounds >= 60)].sort_values(["year", "map_name"])
    P("\nVitality 各图 T 方习惯（下包率 / A 点 / B 点 / 下包时间中位数 / 手枪局胜率）:")
    for r in v.itertuples():
        P(f"  {int(r.year)} {r.map_name}: 下包 {r.plant_pct:.0%}，A {r.site_a_pct:.0%} / B {r.site_b_pct:.0%}，{r.plant_time_median:.0f}s，手枪局 {r.pistol_win_pct:.0%}")
    txt = "\n".join(L)
    open(f"{C}/p3_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
