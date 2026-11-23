"""论文评估：消融（战队向量）、习惯保真度、风格地图、走位模型评估。

用法：python p11_paper_eval.py [events|embed|motion|all]
输出：$CS2_DATA/paper/
    table_team_habits.csv / .md     战队习惯保真度（基线 vs 战队版；训练集内 / 训练集外）
    table_player_style.md           选手风格相关系数（基线 vs 战队版）
    fig_player_embedding.png        选手风格地图（按角色上色）
    fig_team_embedding.png          战队风格地图
    fig_player_years.png            明星选手 2023→2026 风格变化
    table_motion.md                 走位模型：击杀位置误差、热力图 JS 散度（对比两个基线）
    results.md                      所有结果汇总
"""
import paths
import json
import os
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import p4_model as p4
import p7_event_model as em
import p10_motion as mo

BASE = paths.MODELS
OUT = os.path.join(paths.DATA, "paper")
C = em.C
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROLE_COL = {"狙击手": "#e45756", "突破手": "#f58518", "自由人": "#54a24b", "辅助": "#4c78a8", "步枪手": "#9d9d9d"}


def log(*a):
    print(*a, flush=True)


def gen_events(G, A, idx, seed=0, bs=1024):
    """以真实回合条件（阵容/装备/比分/战队）各生成一次事件。"""
    g = torch.Generator(device=em.DEV).manual_seed(seed)
    keys = ("m", "p", "py", "tm", "tmy", "side", "eq", "present", "rctx")
    outs = []
    for i in range(0, len(idx), bs):
        b = em.to_dev({k: A[k] for k in keys}, idx[i:i + bs])
        e, _ = em.sample_rounds(G.model, b, G.place_mask, g)
        outs.append(e)
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def group_stats(E, A, idx, groups, id2place):
    """每组（战队-年份-地图，作为 T 方）的下包率、A 点占比、下包时间；作为 CT 方的击杀点位分布。"""
    rows = []
    for key, gi in groups.items():
        loc = np.searchsorted(idx, gi)
        ty, site, t = E["type"][loc], E["site"][loc], E["t"][loc]
        planted = (ty == 3).any(1)
        s = site[ty == 3]
        pt = t[ty == 3]
        rows.append({"group": key, "n": len(gi), "plant_rate": planted.mean(), "a_share": (s == 1).mean() if len(s) else np.nan,
                     "plant_time": float(np.median(pt)) * em.TBIN if len(pt) else np.nan})
    return pd.DataFrame(rows).set_index("group")


def ct_place_js(E, A, idx, groups_ct, n_place):
    out = {}
    for key, (gi, ct_first) in groups_ct.items():
        loc = np.searchsorted(idx, gi)
        kill = E["type"][loc] == 2
        att = E["a"][loc]
        is_ct_att = np.where(att < 10, np.take_along_axis(A["side"][gi], np.clip(att, 0, 9), 1), -1) == 1
        pl = E["apl"][loc][kill & is_ct_att]
        out[key] = np.bincount(pl, minlength=n_place) + 0.5
    return out


def js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def eval_events():
    maps, rd, sl, ev = em.load_tables()
    TT = os.environ.get("CS2_EVAL_TEAM_TAG", "team")       # 对比哪个战队版：team 或 team2
    Gt = em.Generator.load(f"{BASE}/event_{TT}")
    Gb = em.Generator.load(f"{BASE}/event")
    A = em.build_arrays(rd, sl, ev, Gt.V)
    side0 = A["side"][:, 0]
    t_slug = np.where(side0 == 0, rd.team1_slug.values, rd.team2_slug.values)
    ct_slug = np.where(side0 == 1, rd.team1_slug.values, rd.team2_slug.values)
    res_rows, md = [], []
    pairs = {}                                    # (划分, 指标) -> {模型: (真实, 生成)}，用于配对 bootstrap
    for split_name, mask, min_n in (("训练集内", rd.split.values == "train", 150), ("训练集外", rd.split.values != "train", 60)):
        idx = np.where(mask)[0]
        keyT = pd.Series(list(zip(t_slug[idx], rd.year.values[idx], rd.map_name.values[idx])), index=idx)
        gT = {k: v.index.values for k, v in keyT.groupby(keyT) if len(v) >= min_n and isinstance(k[0], str)}
        sel = np.unique(np.concatenate(list(gT.values()))) if gT else np.array([], int)
        if not len(sel):
            continue
        real = group_stats({k: A[k][sel] for k in ("type", "site", "t", "a", "apl")}, A, sel, gT, None)
        row = {"划分": split_name, "组数": len(gT)}
        for tag, G in (("基线", Gb), ("战队版", Gt)):
            E = gen_events(G, A, sel)
            gen = group_stats(E, A, sel, gT, None)
            j = real.join(gen, lsuffix="_real", rsuffix="_gen")
            for m in ("plant_rate", "a_share", "plant_time"):
                pairs.setdefault((split_name, m), {})[tag] = j[[f"{m}_real", f"{m}_gen"]]
                ok = j[[f"{m}_real", f"{m}_gen"]].dropna()
                row[f"{tag} {m} 相关"] = round(float(np.corrcoef(ok.iloc[:, 0], ok.iloc[:, 1])[0, 1]), 3)
                row[f"{tag} {m} MAE"] = round(float((ok.iloc[:, 0] - ok.iloc[:, 1]).abs().mean()), 3)
            # CT 方击杀点位分布
            keyC = pd.Series(list(zip(ct_slug[sel], rd.year.values[sel], rd.map_name.values[sel])), index=sel)
            gC = {k: (v.index.values, None) for k, v in keyC.groupby(keyC) if len(v) >= min_n and isinstance(k[0], str)}
            rc = ct_place_js({k: A[k][sel] for k in ("type", "a", "apl")}, A, sel, gC, len(G.V.places) + 1)
            gc = ct_place_js(E, A, sel, gC, len(G.V.places) + 1)
            row[f"{tag} CT击杀点位 JS"] = round(float(np.mean([js(rc[k], gc[k]) for k in rc])), 4)
            log(split_name, tag, {k: v for k, v in row.items() if k.startswith(tag)})
        res_rows.append(row)
    tab = pd.DataFrame(res_rows)
    tab.to_csv(f"{OUT}/table_team_habits_{TT}.csv", index=False)
    md.append("## 战队习惯保真度（生成 vs 真实，按「战队-年份-地图」分组）\n")
    md.append("相关系数越高越好，MAE / JS 散度越低越好。plant_rate=下包率，a_share=A 点占比，plant_time=下包时间（秒）。\n")
    md.append(tab.T.to_markdown())
    # 配对 bootstrap：同一批分组重抽样 1000 次，看「战队版 − 基线」的相关系数差
    rng = np.random.default_rng(0)
    md.append("\n## 配对 bootstrap（1000 次，按分组重抽样）：战队版 − 基线 的相关系数差\n")
    md.append("| 划分 | 指标 | 差值 | 95% 置信区间 | 差值 > 0 的比例 |\n|---|---|---|---|---|")
    boot_rows = []
    for (sp, m), d in pairs.items():
        jb = d["基线"].join(d["战队版"], lsuffix="_b", rsuffix="_t").dropna()
        rb, gb, rt, gt = (jb.iloc[:, k].values for k in range(4))
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        diffs = []
        for _ in range(1000):
            i = rng.integers(0, len(jb), len(jb))
            diffs.append(corr(rt[i], gt[i]) - corr(rb[i], gb[i]))
        diffs = np.array([x for x in diffs if not np.isnan(x)])
        point = corr(rt, gt) - corr(rb, gb)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot_rows.append((sp, m, point, lo, hi, float((diffs > 0).mean())))
        md.append(f"| {sp} | {m} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {(diffs > 0).mean():.1%} |")
    pd.DataFrame(boot_rows, columns=["划分", "指标", "差值", "下限", "上限", "大于0比例"]).to_csv(f"{OUT}/table_bootstrap_{TT}.csv", index=False)
    return "\n".join(md)


def eval_embed():
    Gt = em.Generator.load(f"{BASE}/event_team")
    V = Gt.V
    roles = pd.read_parquet(f"{C}/player_roles.parquet")
    names = json.load(open(f"{BASE}/event_team/names.json"))
    main_role = roles.groupby("steamid").role.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
    rounds = roles.groupby("steamid").rounds.sum().to_dict()
    from sklearn.manifold import TSNE
    W = Gt.model.pe.weight.detach().float().cpu().numpy()
    ids = [(s, i) for s, i in V.p.items() if rounds.get(s, 0) >= 800]
    X = np.stack([W[i] for _, i in ids])
    Z = TSNE(n_components=2, perplexity=25, random_state=0, init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(13, 11))
    for (s, _), z in zip(ids, Z):
        r = main_role.get(s, "步枪手")
        ax.scatter(z[0], z[1], c=ROLE_COL.get(r, "#999"), s=40, alpha=0.8)
    top = sorted(ids, key=lambda x: -rounds.get(x[0], 0))[:60]
    pos = {s: z for (s, _), z in zip(ids, Z)}
    for s, _ in top:
        ax.text(pos[s][0], pos[s][1], names.get(s, s), fontsize=9)
    for r, c in ROLE_COL.items():
        ax.scatter([], [], c=c, label=r)
    ax.legend(fontsize=12)
    ax.set_title("选手风格地图（事件模型学到的选手向量，t-SNE；颜色=数据统计得到的角色）", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_player_embedding.png", dpi=130)
    # 角色可分性：用学到的向量做 kNN 预测角色
    y = np.array([main_role.get(s, "步枪手") for s, _ in ids])
    D = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn5 = np.argsort(D, 1)[:, :5]
    pred = [Counter(y[n]).most_common(1)[0][0] for n in nn5]
    acc = float(np.mean(np.array(pred) == y))
    base = float(max(Counter(y).values()) / len(y))
    # 战队
    Tw = Gt.model.tme.weight.detach().float().cpu().numpy()
    tids = list(V.team.items())
    ZT = TSNE(n_components=2, perplexity=min(15, len(tids) - 1), random_state=0, init="pca").fit_transform(np.stack([Tw[i] for _, i in tids]))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(ZT[:, 0], ZT[:, 1], c="#4c78a8", s=50)
    for (slug, _), z in zip(tids, ZT):
        ax.text(z[0], z[1], slug, fontsize=10)
    ax.set_title("战队风格地图（战队向量，t-SNE）", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_team_embedding.png", dpi=130)
    # 明星选手逐年风格变化（选手向量 + 年份修正）
    PYw = Gt.model.pye.weight.detach().float().cpu().numpy()
    stars = ["ZywOo", "donk", "m0NESY", "ropz", "NiKo", "s1mple", "sh1ro", "KSCERATO"]
    inv = {}
    for s, n in names.items():
        if n in stars and s in V.p and rounds.get(s, 0) > rounds.get(inv.get(n), 0):
            inv[n] = s
    pts, labs = [], []
    for n, s in inv.items():
        for y in (2023, 2024, 2025, 2026):
            j = V.ipy(s, y)
            if j:
                pts.append(W[V.ip(s)] + PYw[j])
                labs.append((n, y))
    if pts:
        allX = np.concatenate([X, np.stack(pts)])
        ZA = TSNE(n_components=2, perplexity=25, random_state=0, init="pca").fit_transform(allX)
        Zs = ZA[len(X):]
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(ZA[:len(X), 0], ZA[:len(X), 1], c="#dddddd", s=15)
        for n in inv:
            ii = [k for k, (nn_, _) in enumerate(labs) if nn_ == n]
            ax.plot(Zs[ii, 0], Zs[ii, 1], "-o", lw=2, label=n)
            for k in ii:
                ax.text(Zs[k, 0], Zs[k, 1], str(labs[k][1]), fontsize=8)
        ax.legend()
        ax.set_title("明星选手的风格逐年变化（选手向量 + 年份修正，t-SNE）", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_player_years.png", dpi=130)
    return (f"## 选手风格向量\n\n用 5 近邻在学到的选手向量上预测「统计角色」：准确率 **{acc:.1%}**（多数类基线 {base:.1%}，"
            f"{len(ids)} 名选手）。向量训练时从未见过角色标签，说明角色信息是模型自己从比赛事件中学出来的。\n\n"
            f"图：fig_player_embedding.png、fig_team_embedding.png、fig_player_years.png\n")


def eval_motion(n_rounds=240):
    maps, rd, V, A, B, tracks = mo.load_data()
    M = mo.Motion.load()
    rng = np.random.default_rng(0)
    has_walk = np.array([m in M.walk for m in em.MAPS])
    te = np.where((rd.split.values == "test") & (A["trow"] >= 0) & (A["nsteps"] >= 30) & has_walk[A["m"]])[0]
    pick = np.sort(rng.choice(te, size=min(n_rounds, len(te)), replace=False))
    keys = ("m", "rctx", "tm", "tmy", "p", "py", "side", "eq", "type", "t1", "v", "a", "vpl", "site", "vxy", "axy")
    err = {"模型": [], "原地不动": [], "套用其他回合": []}
    H = {}
    t0 = time.time()
    by_map = {m: np.where(A["m"][te] == m)[0] for m in np.unique(A["m"][te])}
    for n, i in enumerate(pick):
        real = np.asarray(tracks[A["trow"][i]], dtype=np.float32)[:A["nsteps"][i]]
        ns = len(real)
        death = [None] * 10
        kills = []
        for j in range(1, em.MAXEV):
            if A["type"][i, j] == 2 and A["v"][i, j] < 10:
                death[A["v"][i, j]] = int(A["t1"][i, j]) + 1
                kills.append((int(A["v"][i, j]), int(A["t1"][i, j])))
        cond = {k: torch.as_tensor(A[k][i:i + 1], device=mo.DEV) for k in keys}
        mname = em.MAPS[A["m"][i]]
        gen = M.generate(cond, mname, ns, death, seed=int(i), init=real[0, :, :3])
        other = te[by_map[A["m"][i]]]
        o = other[rng.integers(len(other))]
        oth = np.asarray(tracks[A["trow"][o]], dtype=np.float32)
        for v, t in kills:
            if t >= ns or real[t, v, 3] <= 0 and t > 0 and real[t - 1, v, 3] <= 0:
                continue
            rp = real[min(t, ns - 1), v, :2]
            err["模型"].append(float(np.hypot(*(gen[min(t, ns - 1), v, :2] - rp))))
            err["原地不动"].append(float(np.hypot(*(real[0, v, :2] - rp))))
            same_side = [s for s in range(10) if A["side"][o, s] == A["side"][i, v]]
            s2 = same_side[rng.integers(len(same_side))] if same_side else v
            err["套用其他回合"].append(float(np.hypot(*(oth[min(t, 159), s2, :2] - rp))))
        b = B[A["m"][i]]
        for name, tr in (("real", real), ("gen", gen), ("other", oth[:ns])):
            for side in (0, 1):
                al = (tr[..., 3] > 0) & (A["side"][i][None, :] == side)
                h, _, _ = np.histogram2d(tr[..., 0][al], tr[..., 1][al], bins=48, range=[[b[0], b[1]], [b[2], b[3]]])
                H.setdefault((mname, side, name), np.zeros((48, 48)))
                H[(mname, side, name)] += h
        if (n + 1) % 40 == 0:
            log(f"  走位评估 {n + 1}/{len(pick)}  {time.time() - t0:.0f}s")
    jsd = {"模型": [], "套用其他回合": []}
    for (m, side, name), h in H.items():
        if name == "real":
            jsd["模型"].append(js(h.ravel() + 1, H[(m, side, "gen")].ravel() + 1))
            jsd["套用其他回合"].append(js(h.ravel() + 1, H[(m, side, "other")].ravel() + 1))
    md = ["## 走位模型（盲测回合，用真实剧本和真实开局站位为条件）\n",
          "| 方法 | 击杀时刻受害者位置误差（中位数，游戏单位） | 站位热力图 JS 散度（越低越像） |", "|---|---|---|"]
    for k in err:
        md.append(f"| {k} | {np.median(err[k]):.0f} | {np.mean(jsd[k]) if k in jsd else float('nan'):.4f} |")
    md.append(f"\n共 {len(pick)} 个盲测回合、{len(err['模型'])} 次击杀。地图宽度约 4000 单位，跑步速度 250 单位/秒。")
    open(f"{OUT}/table_motion.md", "w").write("\n".join(md))
    return "\n".join(md)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    parts = []
    t0 = time.time()
    if what in ("events", "all", "events_embed"):
        parts.append(eval_events())
    if what in ("embed", "all", "events_embed", "embed_motion"):
        parts.append(eval_embed())
    if what in ("motion", "all", "embed_motion"):
        parts.append(eval_motion())
    txt = f"# 论文实验结果（{time.strftime('%F %T')}，用时 {time.time() - t0:.0f}s）\n\n" + "\n\n".join(parts)
    open(f"{OUT}/results_{what}_{os.environ.get('CS2_EVAL_TEAM_TAG', 'team')}.md", "w").write(txt)
    print(txt)
