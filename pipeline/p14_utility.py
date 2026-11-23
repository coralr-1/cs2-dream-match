"""V8 道具：从原始录像提取每回合的烟雾弹、燃烧弹、闪光弹、手雷（谁扔的、什么时候扔、从哪扔、落在哪、持续多久）。

输入：clean/{rounds,round_slots}.parquet + 原始表 smokes / infernos / grenades
输出：clean/utility.parquet，每行一颗道具：
  map_uid, real_round, type(smoke/molotov/flash/he), slot, steamid,
  t_throw, t_start, t_end（相对冻结时间结束的秒数）, x0, y0（出手位置）, x, y, z（落点 / 爆点）

用法（在有原始数据的机器上）：python p14_utility.py
"""
import paths
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from p2_features import read, PROJECTILES, HELD
from p6_events import TICKRATE

C = paths.CLEAN
TYPE_OF = {"smokes": "smoke", "molotovs": "molotov", "flashes": "flash", "hes": "he"}
SMOKE_S, FIRE_S = 18.0, 7.0          # 表里缺结束时间时的默认持续秒数（CS2 烟雾约 18 秒、火约 7 秒）


def projectiles(path, keep, fmap):
    """每个飞行实体：类型、投掷者、出手 tick/位置、落地（最后一帧）tick/位置。"""
    gr = read(path, "grenades", ["round_num", "thrower_steamid", "grenade_type", "entity_id", "tick", "X", "Y", "Z"], fmap)
    gr = gr[gr.round_num.isin(keep)].sort_values("tick")
    g = gr.groupby(["round_num", "entity_id", "grenade_type"])
    ent = g.agg(thrower=("thrower_steamid", "first"), t0=("tick", "min"), t1=("tick", "max"),
                x0=("X", "first"), y0=("Y", "first"), x1=("X", "last"), y1=("Y", "last"), z1=("Z", "last")).reset_index()
    ent = ent.rename(columns={"grenade_type": "gtype"})
    proj = ent[ent.gtype.str.endswith("Projectile")].copy()
    proj["col"] = proj.gtype.map(PROJECTILES)
    generic = proj[proj.col.isna()].dropna(subset=["thrower"]).sort_values("t0")
    held = ent[ent.gtype.isin(list(HELD))].dropna(subset=["thrower"])
    if len(generic) and len(held):                 # 旧版 demo：用投掷者手里同时消失的道具判断类型
        h = held.assign(col=held.gtype.map(HELD))[["round_num", "thrower", "t1", "col"]].rename(columns={"t1": "t0"}).sort_values("t0")
        mt = pd.merge_asof(generic.drop(columns="col"), h, on="t0", by=["thrower", "round_num"], direction="nearest", tolerance=32)
        proj = pd.concat([proj[proj.col.notna()], mt], ignore_index=True)
    return proj.dropna(subset=["col"])


def part_utility(task):
    path, rds, slots = task
    try:
        fmap = {float(s): s for s in slots.steamid.unique()}
        keep = set(rds.part_round_num)
        proj = projectiles(path, keep, fmap)
        rows = []
        # 闪光 / 手雷：飞行实体最后一帧就是爆点
        for p in proj[proj.col.isin(["flashes", "hes"])].itertuples(index=False):
            rows.append({"round_num": p.round_num, "type": TYPE_OF[p.col], "steamid": p.thrower, "tick_throw": p.t0, "tick_start": p.t1,
                         "tick_end": p.t1, "x0": p.x0, "y0": p.y0, "x": p.x1, "y": p.y1, "z": p.z1})
        # 烟雾：smokes 表给落点和持续时间；出手时刻/位置用同一实体的飞行轨迹
        sm = read(path, "smokes", ["round_num", "entity_id", "start_tick", "end_tick", "thrower_steamid", "thrower_X", "thrower_Y", "X", "Y", "Z"], fmap)
        sm = sm[sm.round_num.isin(keep)]
        sp = proj[proj.col == "smokes"].set_index(["round_num", "entity_id"])
        for s in sm.itertuples(index=False):
            k = (s.round_num, s.entity_id)
            pr = sp.loc[k] if k in sp.index else None
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            t0 = pr.t0 if pr is not None else s.start_tick - 2 * TICKRATE
            x0, y0 = (pr.x0, pr.y0) if pr is not None else (s.thrower_X, s.thrower_Y)
            end = s.end_tick if pd.notna(s.end_tick) else s.start_tick + SMOKE_S * TICKRATE
            rows.append({"round_num": s.round_num, "type": "smoke", "steamid": s.thrower_steamid, "tick_throw": t0, "tick_start": s.start_tick,
                         "tick_end": end, "x0": x0, "y0": y0, "x": s.X, "y": s.Y, "z": s.Z})
        # 燃烧：infernos 表给起火位置和持续时间；出手信息按同一投掷者、起火前最近的燃烧瓶飞行实体匹配
        inf = read(path, "infernos", ["round_num", "start_tick", "end_tick", "thrower_steamid", "thrower_X", "thrower_Y", "X", "Y", "Z"], fmap)
        inf = inf[inf.round_num.isin(keep)].dropna(subset=["start_tick"]).sort_values("start_tick")
        mp = proj[proj.col == "molotovs"].dropna(subset=["thrower"]).sort_values("t1")
        inf = inf.astype({"round_num": "int64", "start_tick": "float64"})                 # merge_asof 要求两边键类型一致
        mp = mp.astype({"round_num": "int64", "t1": "float64", "t0": "float64"})
        if len(inf):
            if len(mp):
                m = pd.merge_asof(inf.rename(columns={"start_tick": "t1", "thrower_steamid": "thrower"}),
                                  mp[["round_num", "thrower", "t0", "t1", "x0", "y0"]], on="t1", by=["round_num", "thrower"],
                                  direction="backward", tolerance=3 * TICKRATE)
            else:
                m = inf.rename(columns={"start_tick": "t1", "thrower_steamid": "thrower"}).assign(t0=np.nan, x0=np.nan, y0=np.nan)
            for s in m.itertuples(index=False):
                end = s.end_tick if pd.notna(s.end_tick) else s.t1 + FIRE_S * TICKRATE
                rows.append({"round_num": s.round_num, "type": "molotov", "steamid": s.thrower,
                             "tick_throw": s.t0 if pd.notna(s.t0) else s.t1 - 1.5 * TICKRATE, "tick_start": s.t1, "tick_end": end,
                             "x0": s.x0 if pd.notna(s.x0) else s.thrower_X, "y0": s.y0 if pd.notna(s.y0) else s.thrower_Y, "x": s.X, "y": s.Y, "z": s.Z})
        if not rows:
            return pd.DataFrame(), None
        u = pd.DataFrame(rows)
        out = []
        for r in rds.itertuples(index=False):
            x = u[(u.round_num == r.part_round_num) & (u.tick_start <= r.end + 5 * TICKRATE)].copy()
            if not len(x):
                continue
            sl = slots[slots.real_round == r.real_round]
            slot_of = dict(zip(sl.steamid, sl.slot))
            x["slot"] = x.steamid.map(slot_of).fillna(-1).astype(int)
            for c in ("throw", "start", "end"):
                x[f"t_{c}"] = (x[f"tick_{c}"] - r.freeze_end) / TICKRATE
            x["map_uid"], x["real_round"] = r.map_uid, r.real_round
            out.append(x)
        if not out:
            return pd.DataFrame(), None
        df = pd.concat(out, ignore_index=True)
        return df[["map_uid", "real_round", "type", "slot", "steamid", "t_throw", "t_start", "t_end", "x0", "y0", "x", "y", "z"]], None
    except Exception as e:
        return pd.DataFrame(), f"{path}: {type(e).__name__}: {e}"[:300]


def main(limit=None):
    t0 = time.time()
    rd = pd.read_parquet(f"{C}/rounds.parquet")
    slots = pd.read_parquet(f"{C}/round_slots.parquet")
    slots["steamid"] = slots.steamid.astype(str)
    sl_by_map = dict(tuple(slots.groupby("map_uid")))
    tasks = [(p, g[["map_uid", "real_round", "part_round_num", "freeze_end", "end"]], sl_by_map.get(g.map_uid.iat[0], slots.iloc[:0]))
             for p, g in rd.groupby("part_path")]
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} 个分段……", flush=True)
    parts, errs = [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for n, (df, err) in enumerate(ex.map(part_utility, tasks, chunksize=2), start=1):
            if len(df):
                parts.append(df)
            if err:
                errs.append(err)
            if n % 500 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    u = pd.concat(parts, ignore_index=True)
    for c in ("t_throw", "t_start", "t_end", "x0", "y0", "x", "y", "z"):
        u[c] = u[c].astype("float32")
    u["slot"] = u.slot.astype("int8")
    u.to_parquet(f"{C}/utility.parquet", index=False)
    per = u.groupby(["map_uid", "real_round"]).size()
    print(f"道具 {len(u)} 颗：{u.type.value_counts().to_dict()}；覆盖 {len(per)} 回合，每回合 {per.mean():.1f} 颗；"
          f"出错分段 {len(errs)}；没对上位置号 {(u.slot < 0).mean():.1%}；用时 {time.time() - t0:.0f}s")
    for e in errs[:5]:
        print("  错误:", e)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
