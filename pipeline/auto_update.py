"""V5 自动更新：把新数据（Kaggle 同格式 parquet 的数据集文件夹）放进 inbox/，运行本脚本或交给定时任务即可。

流程：
  1 找出 inbox 里尚未处理、且 15 分钟内没有再变化（已经拷贝完）的数据集文件夹
  2 P0 索引 + P1 清洗（只针对新文件夹）
  3 只对新地图提取：P2 选手回合特征、V1 事件序列、P3 站位快照与下包、V3a 走位数组
  4 合并进主数据表，重算选手/战队年度汇总（P2）和习惯画像（P3）
  5 把关：用「不含新数据」训练的候选 P4，与现役 P4 在新地图上比较整场预测；候选不明显更差（对数损失不高出 0.02）才继续
  6 用全部数据训练新的 P4、战队版事件模型，并从现役权重继续训练走位模型；现役模型归档到 models_archive/，然后替换
  每一步都写进 update_log.txt，处理状态记在 update_state.json；任何一步失败都不会动现役模型

路径（环境变量）：CS2_HOME（默认为本脚本上一级目录）、CS2_CLEAN（默认 $CS2_HOME/clean）、CS2_MODELS（默认 $CS2_HOME/models）、
                 CS2_INBOX（默认 $CS2_HOME/inbox）
参数：--force 不等 15 分钟；--quick 极少轮数（只用于测试流程）；--no-motion 跳过走位模型继续训练；--no-gate 跳过把关
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))
HOME = os.environ.get("CS2_HOME", os.path.dirname(PIPE))
CLEAN = os.environ.get("CS2_CLEAN", f"{HOME}/clean")
MODELS = os.environ.get("CS2_MODELS", f"{HOME}/models")
INBOX = os.environ.get("CS2_INBOX", f"{HOME}/inbox")
STAGE = f"{HOME}/staging"
STATE = f"{HOME}/update_state.json"
LOG = f"{HOME}/update_log.txt"
os.environ["CS2_CLEAN"], os.environ["CS2_MODELS"] = CLEAN, MODELS          # 让各模块读写主数据表
GATE_TOL = 0.02
WORKERS = max(1, (os.cpu_count() or 4) - 1)


def log(msg):
    line = time.strftime("[%F %T] ") + str(msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"processed": [], "history": []}


def save_state(st):
    json.dump(st, open(STATE, "w"), ensure_ascii=False, indent=1)


def run_py(script, args=(), env=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run([sys.executable, f"{PIPE}/{script}", *args], env=e, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    if r.returncode:
        raise RuntimeError(f"{script} 失败：\n{tail}")
    return tail


# ======================================================================== 1 找新数据
def new_folders(st, force=False):
    if not os.path.isdir(INBOX):
        return []
    out = []
    for d in sorted(os.listdir(INBOX)):
        p = f"{INBOX}/{d}"
        if not os.path.isdir(p) or d in st["processed"] or d.startswith("."):
            continue
        files = glob.glob(f"{p}/**/rounds.parquet", recursive=True)
        if not files:
            continue
        newest = max(os.path.getmtime(f) for f in glob.glob(f"{p}/**/*", recursive=True) if os.path.isfile(f))
        if not force and time.time() - newest < 15 * 60:
            log(f"{d}：最近 15 分钟内还有文件变化，可能还在拷贝，下次再处理")
            continue
        out.append(d)
    return out


# ======================================================================== 2 索引 + 清洗
def stage_clean(folder):
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(f"{STAGE}/clean")
    run_py("p0_index.py", env={"CS2_ROOT": INBOX, "CS2_RAW_GLOB": folder, "CS2_INDEX": f"{STAGE}/_index"})
    t = run_py("p1_clean.py", env={"CS2_ROOT": INBOX, "CS2_INDEX": f"{STAGE}/_index", "CS2_CLEAN": f"{STAGE}/clean"})
    log("P1 清洗：" + " | ".join(l for l in t.splitlines() if "比分合法" in l or "合并后地图" in l))


# ======================================================================== 3 只对新地图提取
def extract_new():
    import p2_features as p2
    import p3_profiles as p3
    import p6_events as p6
    import p9_tracks as p9
    sc = f"{STAGE}/clean"
    maps = pd.read_parquet(f"{sc}/maps.parquet")
    rd = pd.read_parquet(f"{sc}/rounds.parquet")
    rp = pd.read_parquet(f"{sc}/round_players.parquet")
    old = set(pd.read_parquet(f"{CLEAN}/maps.parquet", columns=["map_uid"]).map_uid) if os.path.exists(f"{CLEAN}/maps.parquet") else set()
    new = set(maps.map_uid) - old
    maps, rd, rp = maps[maps.map_uid.isin(new)], rd[rd.map_uid.isin(new)], rp[rp.map_uid.isin(new)]
    rp = rp.assign(steamid=rp.steamid.astype(str))
    log(f"新地图 {len(new)} 张（{len(rd)} 回合）；已存在而跳过 {len(set(pd.read_parquet(f'{sc}/maps.parquet').map_uid) & old)} 张")
    if not new:
        return None
    slots = pd.concat([p6.slot_table(g) for _, g in rp.groupby("map_uid")], ignore_index=True)
    sl_by, rp_by = dict(tuple(slots.groupby("map_uid"))), dict(tuple(rp.groupby("map_uid")))
    T2, T6, T9, T3 = [], [], [], []
    for path, g in rd.groupby("part_path"):
        u = g.map_uid.iat[0]
        T2.append((path, g[["map_uid", "real_round", "part_round_num", "freeze_end", "winner_team"]], rp_by[u]))
        T6.append((path, g[["map_uid", "real_round", "part_round_num", "freeze_end", "end", "reason", "winner_team"]], sl_by[u]))
        T9.append((path, g[["map_uid", "real_round", "freeze_end", "end"]], sl_by[u][["real_round", "steamid", "slot"]]))
        T3.append((path, g[["map_uid", "real_round", "part_round_num", "freeze_end", "end"]]))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        r2 = list(ex.map(p2.part_features, T2))
        r6 = list(ex.map(p6.part_events, T6))
        r9 = list(ex.map(p9.part_tracks, T9))
        r3 = list(ex.map(p3.part_positions, T3))
    errs = [e for x in r2 + r6 + r9 for e in [x[1]] if e] + [x[2] for x in r3 if x[2]]
    for e in errs[:5]:
        log(f"  提取出错：{e}")
    pr = pd.concat([x[0] for x in r2], ignore_index=True).merge(maps[["map_uid", "year", "event_id", "map_name", "team1_slug", "team2_slug"]], on="map_uid", how="left")
    pr["team_slug"] = np.where(pr.team == 1, pr.team1_slug, pr.team2_slug)
    pr = pr.drop(columns=["team1_slug", "team2_slug"])
    ev = pd.concat([x[0] for x in r6], ignore_index=True).drop(columns=["tick"])
    for c in ("a_slot", "v_slot", "as_slot"):
        ev[c] = ev[c].fillna(-1).astype(int)
    tracks = [t for x in r9 for t in x[0]]
    pos = pd.concat([x[0] for x in r3], ignore_index=True)
    pos = pos.merge(rp[["map_uid", "real_round", "steamid", "team"]], on=["map_uid", "real_round", "steamid"], how="inner")
    pos = pos.merge(maps[["map_uid", "map_name", "year", "team1_slug", "team2_slug"]], on="map_uid", how="left")
    pos["team_slug"] = np.where(pos.team == 1, pos.team1_slug, pos.team2_slug)
    pos = pos.drop(columns=["team1_slug", "team2_slug"])
    bombs = pd.concat([x[1] for x in r3], ignore_index=True)
    return {"maps": maps, "rounds": rd, "round_players": rp, "round_slots": slots, "player_rounds": pr, "events": ev,
            "positions": pos, "round_bomb": bombs, "tracks": tracks, "new": sorted(new)}


# ======================================================================== 4 合并 + 重算汇总
def merge(N):
    """先把合并结果全部写到 staging/merged/，全部成功后再一次性替换进主数据目录，避免中途失败留下半更新的数据。"""
    out_dir = f"{STAGE}/merged"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    os.makedirs(CLEAN, exist_ok=True)
    for name in ("maps", "rounds", "round_players", "round_slots", "player_rounds", "events", "positions", "round_bomb"):
        f = f"{CLEAN}/{name}.parquet"
        df = N[name]
        if os.path.exists(f):
            old = pd.read_parquet(f)
            if "steamid" in old.columns and "steamid" in df.columns:
                old["steamid"], df = old.steamid.astype(str), df.assign(steamid=df.steamid.astype(str))
            df = pd.concat([old[~old.map_uid.isin(set(df.map_uid))], df], ignore_index=True)
        df.to_parquet(f"{out_dir}/{name}.parquet", index=False)
    # 选手表
    rp = pd.read_parquet(f"{out_dir}/round_players.parquet").merge(pd.read_parquet(f"{out_dir}/maps.parquet", columns=["map_uid", "year"]), on="map_uid")
    pl = rp.groupby("steamid").agg(name=("name", lambda s: s.mode().iat[0]), aliases=("name", lambda s: "|".join(sorted(set(map(str, s))))),
                                   n_maps=("map_uid", "nunique"), n_rounds=("real_round", "size"),
                                   years=("year", lambda s: ",".join(str(int(y)) for y in sorted(s.dropna().unique())))).reset_index()
    pl.to_parquet(f"{out_dir}/players.parquet", index=False)
    # 走位数组：旧数组（去掉被替换的地图）+ 新回合
    K = 160
    idx_f, arr_f = f"{CLEAN}/tracks_index.parquet", f"{CLEAN}/tracks.npy"
    old_ix = pd.read_parquet(idx_f) if os.path.exists(idx_f) else pd.DataFrame(columns=["map_uid", "real_round", "n_steps", "row"])
    old_arr = np.load(arr_f, mmap_mode="r") if os.path.exists(arr_f) else np.zeros((0, K, 10, 4), np.float16)
    keep = ~old_ix.map_uid.isin(set(N["new"]))
    rows = old_ix[keep].row.values.astype(int)
    n_old, n_new = len(rows), len(N["tracks"])
    out = np.lib.format.open_memmap(f"{out_dir}/tracks.npy", mode="w+", dtype=np.float16, shape=(n_old + n_new, K, 10, 4))
    for i in range(0, n_old, 5000):
        chunk = rows[i:i + 5000]
        out[i:i + len(chunk)] = old_arr[chunk]
    for j, (u, rr, ns, a) in enumerate(N["tracks"]):
        out[n_old + j] = a
    out.flush()
    del out, old_arr
    ix = pd.concat([old_ix[keep][["map_uid", "real_round", "n_steps"]],
                    pd.DataFrame([(u, rr, ns) for u, rr, ns, _ in N["tracks"]], columns=["map_uid", "real_round", "n_steps"])], ignore_index=True)
    ix["row"] = np.arange(len(ix))
    ix.to_parquet(f"{out_dir}/tracks_index.parquet", index=False)
    # 全部写好后再替换
    for fn in os.listdir(out_dir):
        os.replace(f"{out_dir}/{fn}", f"{CLEAN}/{fn}")
    log(f"合并完成：{len(pd.read_parquet(f'{CLEAN}/maps.parquet'))} 张图，走位数组 {len(ix)} 回合（新增 {n_new}）")
    # 汇总与画像
    import p2_features as p2
    import p3_profiles as p3
    p2.summarize_tables()
    p3.summarize()
    log("已重算选手/战队年度汇总和习惯画像")


# ======================================================================== 5 把关
def map_eval(sim, maps, uids, n=400):
    import zlib
    rows = []
    for r in maps[maps.map_uid.isin(uids) & maps.legal.astype(bool)].itertuples(index=False):
        A = [(s, int(r.year)) for s in str(r.roster1).split(",") if s]
        B = [(s, int(r.year)) for s in str(r.roster2).split(",") if s]
        if len(A) == 5 and len(B) == 5:
            rows.append((sim.simulate(A, B, r.map_name, n=n, seed=zlib.crc32(r.map_uid.encode()))["p_win_A"], float(r.score1 > r.score2)))
    if not rows:
        return None, 0
    p, y = np.array(rows).T
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), len(rows)


def gate(new_uids, quick):
    import p4_model as p4
    ex = f"{STAGE}/exclude.txt"
    open(ex, "w").write("\n".join(new_uids))
    cand = f"{STAGE}/cand_models"
    env = {"CS2_MODELS": cand, "CS2_FINAL": "1", "CS2_EXCLUDE_MAPS": ex, "CS2_CALIB_SRC": f"{MODELS}/calib.json"}
    if quick:
        env["CS2_QUICK"] = "1"
    run_py("p4_model.py", ["train"], env)
    maps = pd.read_parquet(f"{CLEAN}/maps.parquet")
    ll_cur, n = map_eval(p4.Simulator.load(MODELS), maps, new_uids)
    ll_cand, _ = map_eval(p4.Simulator.load(cand), maps, new_uids)
    if ll_cur is None:
        log("新数据里没有可评估的完整地图，跳过把关")
        return True, {}
    ok = ll_cand <= ll_cur + GATE_TOL
    log(f"把关（新数据 {n} 张图，整场对数损失）：现役 {ll_cur:.4f}，候选 {ll_cand:.4f} → {'通过' if ok else '不通过，保留现役模型'}")
    return ok, {"maps": n, "logloss_current": ll_cur, "logloss_candidate": ll_cand}


# ======================================================================== 6 重训 + 替换
def retrain_and_promote(quick, motion=True):
    new = f"{STAGE}/new_models"
    shutil.rmtree(new, ignore_errors=True)
    env = {"CS2_MODELS": new, "CS2_FINAL": "1", "CS2_CALIB_SRC": f"{MODELS}/calib.json"}
    if quick:
        env["CS2_QUICK"] = "1"
    log("训练新 P4……")
    run_py("p4_model.py", ["train"], env)
    log("训练新 Ban/Pick 模型……")
    run_py("p12_veto.py", ["train"], env)
    log("训练新事件模型（战队版 v2）……")
    run_py("p7_event_model.py", ["train"], {**env, "CS2_TAG": "team2"})
    os.replace(f"{new}/event_team2", f"{new}/event_team")         # 出片程序读取的是 event_team
    log("重建地图物理（视线 / 可走区域 / 真实对枪位置）……")
    run_py("p13_physics.py", ["build"], env)
    if motion and os.path.exists(f"{MODELS}/motion/motion_model.pt"):
        log("从现役权重继续训练走位模型……")
        run_py("p10_motion.py", ["train"], {**env, "CS2_MOTION_INIT": f"{MODELS}/motion/motion_model.pt", "CS2_MOTION_EPOCHS": "1" if quick else "3"})
    elif os.path.exists(f"{MODELS}/motion"):
        shutil.copytree(f"{MODELS}/motion", f"{new}/motion")
    arch = f"{HOME}/models_archive/{time.strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(MODELS):
        shutil.copytree(MODELS, arch)
    for f in glob.glob(f"{MODELS}/round_model_*.pt"):          # 新 P4 的模型个数可能不同，先清掉旧文件
        os.remove(f)
    for item in os.listdir(new):
        dst = f"{MODELS}/{item}"
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        (shutil.copytree if os.path.isdir(f"{new}/{item}") else shutil.copy)(f"{new}/{item}", dst)
    log(f"新模型已启用；旧模型归档在 {arch}（回退：把归档里的文件复制回 {MODELS}）")
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    lock = f"{HOME}/.update.lock"
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 12 * 3600:
        print("另一个更新正在进行，退出")
        return
    open(lock, "w").write(str(os.getpid()))
    st = load_state()
    try:
        folders = new_folders(st, args.force)
        if not folders:
            print(time.strftime("[%F %T] ") + "inbox 里没有新数据")
            return
        for folder in folders:
            log(f"===== 开始处理 {folder} =====")
            t0 = time.time()
            rec = {"folder": folder, "start": time.strftime("%F %T")}
            try:
                stage_clean(folder)
                N = extract_new()
                if N is None:
                    rec["result"] = "没有新地图"
                else:
                    merge(N)
                    rec["new_maps"] = len(N["new"])
                    ok, metrics = (True, {}) if args.no_gate else gate(N["new"], args.quick)
                    rec["gate"] = metrics
                    if ok:
                        rec["archive"] = retrain_and_promote(args.quick, not args.no_motion)
                        rec["result"] = "已更新模型"
                    else:
                        rec["result"] = "数据已合并，模型未替换（把关不通过）"
                st["processed"].append(folder)
            except Exception as e:
                rec["result"] = f"失败：{e}"
                log(f"处理 {folder} 失败，现役模型未改动：{e}\n{traceback.format_exc()[-1500:]}")
            rec["minutes"] = round((time.time() - t0) / 60, 1)
            st["history"].append(rec)
            save_state(st)
            log(f"===== {folder}：{rec['result']}（{rec['minutes']} 分钟）=====")
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
