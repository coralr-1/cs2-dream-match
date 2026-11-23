"""V3a 走位数据：每回合冻结结束后每秒一帧，10 个位置号的 (x, y, 朝向, 是否存活)。

输入：clean/{rounds,round_slots}.parquet + 原始 ticks
输出：clean/
    tracks.npy             float16 数组 [回合数, 160, 10, 4]：x, y, yaw(度), alive；超出回合时长的帧 alive=-1
    tracks_index.parquet   每行对应 tracks.npy 的一行：map_uid, real_round, n_steps
    map_bounds.json        每张地图坐标范围（0.1%–99.9% 分位），训练时归一化用
    p9_report.txt
"""
import paths
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

C = paths.CLEAN
TICKRATE = 64
K = 160                     # 最多 160 秒


def part_tracks(task):
    path, rds, slots = task
    try:
        want = {}
        for r in rds.itertuples(index=False):
            n = int(min(K, (r.end - r.freeze_end) // TICKRATE + 1))
            for k in range(n):
                want[int(r.freeze_end + k * TICKRATE)] = (r.real_round, k)
        tbl = pq.read_table(f"{path}/ticks.parquet", columns=["tick", "steamid", "X", "Y", "yaw", "health"])
        tbl = tbl.filter(pc.is_in(tbl["tick"], value_set=pa.array(list(want), type=tbl.schema.field("tick").type)))
        df = tbl.to_pandas()
        df["steamid"] = df.steamid.astype(str)
        rk = np.array([want[t] for t in df.tick.values])
        df["real_round"], df["k"] = rk[:, 0], rk[:, 1]
        df = df.merge(slots[["real_round", "steamid", "slot"]], on=["real_round", "steamid"], how="inner")
        out = []
        for r in rds.itertuples(index=False):
            n = int(min(K, (r.end - r.freeze_end) // TICKRATE + 1))
            arr = np.zeros((K, 10, 4), np.float32)
            arr[:, :, 3] = -1
            arr[:n, :, 3] = 0
            g = df[df.real_round == r.real_round]
            if len(g):
                arr[g.k.values, g.slot.values, 0] = g.X.values
                arr[g.k.values, g.slot.values, 1] = g.Y.values
                arr[g.k.values, g.slot.values, 2] = g.yaw.values
                arr[g.k.values, g.slot.values, 3] = (g.health.values > 0).astype(np.float32)
            # 死后（或缺帧）沿用最后已知位置
            for s in range(10):
                last = None
                for k in range(n):
                    if arr[k, s, 3] > 0:
                        last = arr[k, s, :3].copy()
                    elif last is not None:
                        arr[k, s, :3] = last
            out.append((r.map_uid, r.real_round, n, arr.astype(np.float16)))
        return out, None
    except Exception as e:
        return [], f"{path}: {type(e).__name__}: {e}"[:300]


def main(limit=None):
    t0 = time.time()
    rd = pd.read_parquet(f"{C}/rounds.parquet", columns=["map_uid", "real_round", "part_path", "freeze_end", "end"])
    sl = pd.read_parquet(f"{C}/round_slots.parquet", columns=["map_uid", "real_round", "steamid", "slot"])
    sl["steamid"] = sl.steamid.astype(str)
    sl_by_map = dict(tuple(sl.groupby("map_uid")))
    tasks = [(p, g, sl_by_map.get(g.map_uid.iat[0], sl.iloc[:0])) for p, g in rd.groupby("part_path")]
    if limit:
        tasks = tasks[:limit]
    n_rounds = sum(len(t[1]) for t in tasks)
    arr = np.lib.format.open_memmap(f"{C}/tracks.npy", mode="w+", dtype=np.float16, shape=(n_rounds, K, 10, 4))
    idx, errs, i = [], [], 0
    print(f"{len(tasks)} 个分段，{n_rounds} 回合……", flush=True)
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 4)) as ex:
        for n, (res, err) in enumerate(ex.map(part_tracks, tasks, chunksize=2), start=1):
            if err:
                errs.append(err)
            for uid, rr, ns, a in res:
                arr[i] = a
                idx.append((uid, rr, ns))
                i += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    ix = pd.DataFrame(idx, columns=["map_uid", "real_round", "n_steps"])
    ix["row"] = np.arange(len(ix))
    ix.to_parquet(f"{C}/tracks_index.parquet", index=False)
    # 地图坐标范围
    maps = pd.read_parquet(f"{C}/maps.parquet", columns=["map_uid", "map_name"])
    ix = ix.merge(maps, on="map_uid")
    bounds = {}
    rng = np.random.default_rng(0)
    for m, g in ix.groupby("map_name"):
        rows = rng.choice(g.row.values, size=min(len(g), 3000), replace=False)
        a = np.asarray(arr[np.sort(rows)], dtype=np.float32)
        alive = a[..., 3] > 0
        xs, ys = a[..., 0][alive], a[..., 1][alive]
        bounds[m] = [float(np.percentile(xs, 0.1)), float(np.percentile(xs, 99.9)), float(np.percentile(ys, 0.1)), float(np.percentile(ys, 99.9))]
    json.dump(bounds, open(f"{C}/map_bounds.json", "w"), indent=1)
    sample = np.asarray(arr[: min(i, 2000)], dtype=np.float32)
    valid = sample[..., 3] >= 0
    step = np.hypot(np.diff(sample[..., 0], axis=1), np.diff(sample[..., 1], axis=1))
    both = (sample[:, 1:, :, 3] > 0) & (sample[:, :-1, :, 3] > 0)
    L = [f"# V3a 走位数据报告 {time.strftime('%F %T')} 用时 {time.time() - t0:.0f}s",
         f"分段 {len(tasks)}，出错 {len(errs)}；回合 {i}；tracks.npy 形状 {arr.shape}，约 {arr.nbytes / 1e9:.2f} GB",
         f"回合时长（秒）分位数：{ix.n_steps.quantile([.1, .5, .9]).to_dict()}",
         f"抽样：有效帧中存活比例 {float((sample[..., 3] > 0)[valid].mean()):.3f}；存活时每秒位移 分位数 {np.percentile(step[both], [50, 90, 99]).round(1).tolist()}",
         f"地图范围：{json.dumps(bounds)}"]
    for e in errs[:5]:
        L.append(f"  错误: {e}")
    txt = "\n".join(L)
    open(f"{C}/p9_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
