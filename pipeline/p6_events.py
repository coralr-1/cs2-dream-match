"""V1 事件序列：每回合按时间排列的击杀 / 下包 / 回合结束事件，给事件生成模型和视频渲染用。

输入：clean/{rounds,round_players}.parquet + 原始 kills / bomb 表
输出：clean/
    round_slots.parquet  每回合 10 个「位置号」：0–4 是 team1，5–9 是 team2；同一张图里每名选手的位置号尽量固定
    events.parquet       每回合的事件序列（type = kill / plant / end），时间是冻结结束后的秒数
    p6_events_report.txt
"""
import paths
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from p2_features import read

C = paths.CLEAN
TICKRATE = 64


def slot_table(rp_map):
    """同一张图里按出场回合数给每队选手排名，每回合在场选手按排名占位置号。"""
    rank = rp_map.groupby(["team", "steamid"]).size().rename("n").reset_index()
    rank = rank.sort_values(["team", "n", "steamid"], ascending=[True, False, True])
    rank["rank"] = rank.groupby("team").cumcount()
    x = rp_map.merge(rank[["team", "steamid", "rank"]], on=["team", "steamid"]).sort_values(["real_round", "team", "rank"])
    x["pos"] = x.groupby(["real_round", "team"]).cumcount()
    x = x[x.pos < 5].copy()
    x["slot"] = (x.team - 1) * 5 + x.pos
    return x[["map_uid", "real_round", "slot", "steamid", "team", "side", "equip", "armor", "helmet", "defuser"]]


def part_events(task):
    path, rds, slots = task
    try:
        fmap = {float(s): s for s in slots.steamid.unique()}
        keep = set(rds.part_round_num)
        k = read(path, "kills", ["tick", "round_num", "attacker_steamid", "victim_steamid", "assister_steamid", "assistedflash",
                                 "headshot", "weapon", "penetrated", "thrusmoke", "attacker_place", "victim_place",
                                 "attacker_X", "attacker_Y", "victim_X", "victim_Y"], fmap)
        k = k[k.round_num.isin(keep)]
        b = read(path, "bomb", ["tick", "round_num", "event", "steamid", "bombsite", "X", "Y"], fmap)
        b = b[b.round_num.isin(keep) & (b.event.astype(str) == "plant")]
        out = []
        for r in rds.itertuples(index=False):
            sl = slots[slots.real_round == r.real_round]
            slot_of = dict(zip(sl.steamid, sl.slot))
            fe, end = r.freeze_end, r.end
            ev = []
            for e in k[k.round_num == r.part_round_num].itertuples(index=False):
                ev.append({"tick": e.tick, "type": "kill", "a_slot": slot_of.get(e.attacker_steamid, -1), "v_slot": slot_of.get(e.victim_steamid, -1),
                           "as_slot": slot_of.get(e.assister_steamid, -1), "weapon": str(e.weapon).lower(), "hs": bool(e.headshot),
                           "flash_assist": bool(e.assistedflash), "thrusmoke": bool(e.thrusmoke), "penetrated": bool(e.penetrated),
                           "a_place": e.attacker_place, "v_place": e.victim_place,
                           "ax": e.attacker_X, "ay": e.attacker_Y, "vx": e.victim_X, "vy": e.victim_Y})
            for e in b[b.round_num == r.part_round_num].itertuples(index=False):
                ev.append({"tick": e.tick, "type": "plant", "a_slot": slot_of.get(e.steamid, -1),
                           "site": str(e.bombsite).replace("Bombsite", ""), "ax": e.X, "ay": e.Y})
            ev.append({"tick": end, "type": "end", "reason": r.reason, "winner_team": r.winner_team})
            ev.sort(key=lambda x: (x["tick"], {"kill": 0, "plant": 1, "end": 2}[x["type"]]))
            for i, x in enumerate(ev):
                x.update({"map_uid": r.map_uid, "real_round": r.real_round, "seq": i, "time": (x["tick"] - fe) / TICKRATE,
                          "post_end": x["tick"] > end and x["type"] != "end"})
                out.append(x)
        return pd.DataFrame(out), None
    except Exception as e:
        return pd.DataFrame(), f"{path}: {type(e).__name__}: {e}"[:300]


def main(limit=None):
    t0 = time.time()
    rd = pd.read_parquet(f"{C}/rounds.parquet")
    rp = pd.read_parquet(f"{C}/round_players.parquet")
    rp["steamid"] = rp.steamid.astype(str)
    slots = pd.concat([slot_table(g) for _, g in rp.groupby("map_uid")], ignore_index=True)
    slots.to_parquet(f"{C}/round_slots.parquet", index=False)
    sl_by_map = dict(tuple(slots.groupby("map_uid")))
    tasks = [(p, g[["map_uid", "real_round", "part_round_num", "freeze_end", "end", "reason", "winner_team"]], sl_by_map.get(g.map_uid.iat[0], slots.iloc[:0]))
             for p, g in rd.groupby("part_path")]
    if limit:
        tasks = tasks[:limit]
    print(f"{len(tasks)} 个分段……", flush=True)
    parts, errs = [], []
    with ProcessPoolExecutor(max_workers=min(24, os.cpu_count() or 4)) as ex:
        for n, (df, err) in enumerate(ex.map(part_events, tasks, chunksize=2), start=1):
            parts.append(df)
            if err:
                errs.append(err)
            if n % 1000 == 0:
                print(f"  {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    ev = pd.concat(parts, ignore_index=True).drop(columns=["tick"])
    for c in ("a_slot", "v_slot", "as_slot"):
        ev[c] = ev[c].fillna(-1).astype(int)
    ev.to_parquet(f"{C}/events.parquet", index=False)

    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("# V1 事件序列报告", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s")
    P(f"分段 {len(tasks)}，出错 {len(errs)}；事件 {len(ev)} 条：{ev.type.value_counts().to_dict()}")
    for e in errs[:5]:
        P("  错误:", e)
    kl = ev[ev.type == "kill"]
    pre = kl[~kl.post_end]
    P(f"回合结束后的补刀 {int(kl.post_end.sum())} 次（{kl.post_end.mean():.1%}），不计入回合过程")
    P(f"击杀者/受害者未能对上位置号：{(pre.a_slot < 0).mean():.2%} / {(pre.v_slot < 0).mean():.2%}（攻击者为空通常是摔死、炸弹炸死）")
    P(f"同队击杀（误伤/自杀）：{((pre.a_slot >= 0) & ((pre.a_slot // 5) == (pre.v_slot // 5))).mean():.2%}")
    P(f"每回合回合内击杀数：{pre.groupby(['map_uid', 'real_round']).size().describe().round(2).to_dict()}")
    P(f"击杀时间分位数（秒）：{pre.time.quantile([.1, .25, .5, .75, .9]).round(1).to_dict()}")
    P(f"武器前 15：{pre.weapon.value_counts().head(15).to_dict()}")
    P(f"爆头率 {pre.hs.mean():.1%}，穿烟 {pre.thrusmoke.mean():.1%}，穿墙 {pre.penetrated.mean():.1%}")
    P(f"点位为空 {pre.v_place.isna().mean():.2%}；不同点位名 {pre.v_place.nunique()}")
    end = ev[ev.type == "end"]
    P(f"回合结束原因：{end.reason.value_counts().to_dict()}")
    P(f"回合时长（秒）分位数：{end.time.quantile([.1, .5, .9]).round(1).to_dict()}")
    txt = "\n".join(L)
    open(f"{C}/p6_events_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
