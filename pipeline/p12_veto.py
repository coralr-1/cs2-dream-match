"""P12 Ban/Pick（选图）模型：从真实系列赛的打图顺序学习每名选手（进而任意阵容）的选图习惯，并模拟 BO1/BO3/BO5 的 ban/pick。

数据里没有 ban 记录，只有每个系列赛打了哪几张图、按什么顺序。按标准流程
  BO1：A禁 B禁 A禁 B禁 A禁 B禁，剩下的就是比赛地图
  BO3：A禁 B禁 A选 B选 A禁 B禁，剩下的是决胜图（第 1 图 = A 选，第 2 图 = B 选，第 3 图 = 决胜图）
  BO5：A禁 B禁 A选 B选 A选 B选，剩下的是决胜图
没观测到的 ban 和「谁是先手 A」当作隐变量：枚举所有与观测一致的流程，最大化边际似然。

每一步的选择概率（对当前还剩的图做 softmax）：
  选图：γ[时期, 图] + θ_自己(图) − λ_p · θ_对手(图)
  禁图：β[时期, 图] − s_b · (θ_自己(图) − λ_b · θ_对手(图))
θ_阵容 = 5 名选手 (φ[选手] + δ[选手, 年份]) 的平均——和其它模型一样落到选手上，所以自由组合、跨年份阵容也能用。
「时期」= 地图池不变的一段连续赛事；γ/β 是该时期的地图热度（版本强势图），θ 是阵容相对热度的个人偏好。

用法：python p12_veto.py train      （评估 + 用全部数据训练正式模型，写到 $CS2_MODELS/veto）
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import p4_model as p4

C = p4.C
MDIR = os.path.join(p4.MDIR, "veto")
MAPS = p4.MAPS
NM = len(MAPS)
MAP_CN = {"de_ancient": "Ancient", "de_anubis": "Anubis", "de_cache": "Cache", "de_dust2": "Dust2", "de_inferno": "Inferno",
          "de_mirage": "Mirage", "de_nuke": "Nuke", "de_overpass": "Overpass", "de_train": "Train", "de_vertigo": "Vertigo"}
FORMATS = {   # (行动方 0=先手A/1=后手B, 动作 0=禁/1=选)；最后剩下的一张是比赛地图/决胜图
    "bo1": [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0)],
    "bo3": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 0), (1, 0)],
    "bo5": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 1), (1, 1)],
}
MIN_SERIES = 4        # 训练集中至少出现在这么多个系列赛里才有自己的偏好向量
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


# ---------------------------------------------------------------- 数据
def event_pools(maps):
    """每个赛事的 7 图地图池：赛事里出现的图；不足 7 张时（小赛事没打到全部图）用最近一个完整赛事的池子补齐。"""
    order = sorted(maps.event_id.unique())
    seen = maps.groupby("event_id").map_name.agg(lambda x: set(x))
    pools, last = {}, None
    full = [e for e in order if len(seen[e]) >= 7]
    for e in order:
        s = seen[e]
        if len(s) >= 7:
            last = s
            pools[e] = sorted(s)[:7] if len(s) == 7 else sorted(s, key=lambda m: -(maps[(maps.event_id == e) & (maps.map_name == m)].shape[0]))[:7]
        else:
            ref = last if last is not None else seen[full[0]]
            if not s <= ref:                                         # 小赛事打了参考池以外的图：找下一个包含它的完整赛事
                nxt = [seen[f] for f in full if f > e and s <= seen[f]]
                ref = nxt[0] if nxt else ref
            pools[e] = sorted(ref)
    return pools


def pool_periods(maps, pools):
    """把地图池相同的连续赛事合并成「时期」。返回 (时期列表[按时间], 赛事 → 时期编号)。
    时期列表也是界面里的地图池预设（显示时最新的放最前）。"""
    ev = maps.groupby("event_id").agg(year=("year", "first"), name=("event_name", "first")).sort_index()
    periods, of = [], {}
    for e, r in ev.iterrows():
        p = tuple(pools[e])
        if not (periods and periods[-1]["pool"] == p):
            periods.append({"pool": p, "y0": int(r.year), "y1": int(r.year), "events": 0, "first": str(r["name"])})
        periods[-1]["y1"] = int(r.year)
        periods[-1]["events"] += 1
        of[e] = len(periods) - 1
    out = []
    for i, p in enumerate(periods):
        yrs = f"{p['y0']}" if p["y0"] == p["y1"] else f"{p['y0']}–{p['y1']}"
        tag = "最新 · " if i == len(periods) - 1 else ""
        out.append({"label": f"{yrs}（{tag}{p['first']} 起，{p['events']} 个赛事）：" + " / ".join(MAP_CN[m] for m in p["pool"]),
                    "pool": list(p["pool"]), "year": p["y1"], "period": i})
    return out, of


def load_series(maps):
    """每个系列赛：两队阵容、年份、赛事、赛制和观测到的打图顺序。"""
    pools = event_pools(maps)
    periods, period_of = pool_periods(maps, pools)
    rows = []
    for mid, g in maps.sort_values("map_number").groupby("match_id"):
        n = len(g)
        if n > 3 or g.map_number.max() != n:                         # BO5 太少、或者有缺图的系列赛不用
            continue
        r0 = g.iloc[0]
        if not isinstance(r0.roster1, str) or not isinstance(r0.roster2, str):
            continue
        X = r0.roster1.split(",")[:5]
        Y = r0.roster2.split(",")[:5]
        if len(X) < 5 or len(Y) < 5:
            continue
        pool = pools[r0.event_id]
        seq = list(g.map_name)
        if any(m not in pool for m in seq):
            continue
        rows.append({"match_id": mid, "event_id": r0.event_id, "year": int(r0.year), "period": period_of[r0.event_id], "fmt": "bo1" if n == 1 else "bo3",
                     "maps": seq, "pool": pool, "X": X, "Y": Y, "slugX": r0.team1_slug, "slugY": r0.team2_slug})
    return pd.DataFrame(rows), periods


def enumerate_vetoes(fmt, pool, observed):
    """与观测一致的全部流程（相对先后手）。observed：bo1 为 [比赛图]；bo3 为 [第1图, 第2图(, 决胜图)]。
    返回 [(选择序列, 每步可选集合)]。"""
    steps = FORMATS[fmt]
    if fmt == "bo1":
        picks, decider = [], observed[0]
    else:
        picks = observed[:2]
        decider = observed[2] if len(observed) > 2 else None
    out = []

    def rec(k, avail, chosen, avails, pi):
        if k == len(steps):
            if decider is None or avail == {decider}:
                out.append((list(chosen), list(avails)))
            return
        actor, act = steps[k]
        if act == 1:
            m = picks[pi]
            if m in avail:
                rec(k + 1, avail - {m}, chosen + [m], avails + [frozenset(avail)], pi + 1)
            return
        reserved = set(picks[pi:]) | ({decider} if decider else set())
        for m in sorted(avail - reserved):
            rec(k + 1, avail - {m}, chosen + [m], avails + [frozenset(avail)], pi)

    rec(0, set(pool), [], [], 0)
    return out


class Vocab:
    def __init__(self, series, min_n=MIN_SERIES):
        from collections import Counter
        cp, cpy = Counter(), Counter()
        for r in series.itertuples():
            for s in r.X + r.Y:
                cp[s] += 1
                cpy[(s, r.year)] += 1
        self.p = {s: i + 1 for i, s in enumerate(sorted(s for s, c in cp.items() if c >= min_n))}
        self.py = {k: i + 1 for i, k in enumerate(sorted(k for k, c in cpy.items() if c >= min_n))}

    def ip(self, s):
        return self.p.get(str(s), 0)

    def ipy(self, s, y):
        return self.py.get((str(s), int(y)), 0)

    def to_dict(self):
        return {"p": self.p, "py": [[s, y, i] for (s, y), i in self.py.items()]}

    @classmethod
    def from_dict(cls, d):
        v = cls.__new__(cls)
        v.p = d["p"]
        v.py = {(s, int(y)): i for s, y, i in d["py"]}
        return v


# ---------------------------------------------------------------- 模型
class VetoModel(torch.nn.Module):
    def __init__(self, n_p, n_py, n_periods, use_players=True):
        super().__init__()
        self.use_players = use_players
        self.phi = torch.nn.Embedding(n_p + 1, NM)
        self.delta = torch.nn.Embedding(n_py + 1, NM)
        for e in (self.phi, self.delta):
            torch.nn.init.zeros_(e.weight)
        self.gamma = torch.nn.Parameter(torch.zeros(n_periods, NM))
        self.beta = torch.nn.Parameter(torch.zeros(n_periods, NM))
        # λ 限制在 [0,1]、s_b ≥ 0：保证「禁自己弱/对手强的图、选自己强的图」。不加限制时似然略高，
        # 但因为 ban 记录不可见，模型会把一方的「永禁图」算到另一方头上（例：让 Vitality 2025 禁掉它最常打的 Inferno）
        self.lam_p_raw = torch.nn.Parameter(torch.tensor(-1.0))
        self.lam_b_raw = torch.nn.Parameter(torch.tensor(-1.0))
        self.s_b_raw = torch.nn.Parameter(torch.tensor(0.5))

    @property
    def lam_p(self):
        return torch.sigmoid(self.lam_p_raw)

    @property
    def lam_b(self):
        return torch.sigmoid(self.lam_b_raw)

    @property
    def s_b(self):
        return torch.nn.functional.softplus(self.s_b_raw)

    def theta(self, P, PY):
        """P/PY: [N, 5] 选手/选手-年份编号（0 = 没有自己的向量）→ 阵容偏好 [N, 图数]。"""
        if not self.use_players:
            return torch.zeros(P.shape[0], NM)
        v = self.phi(P) * (P > 0).unsqueeze(-1) + self.delta(PY) * (PY > 0).unsqueeze(-1)
        return v.mean(1)

    def logits(self, y, tX, tY):
        """返回 [N, 2(队 X/Y), 2(禁/选), 图数]。"""
        g, b = self.gamma[y], self.beta[y]
        pick = torch.stack([g + tX - self.lam_p * tY, g + tY - self.lam_p * tX], 1)
        ban = torch.stack([b - self.s_b * (tX - self.lam_b * tY), b - self.s_b * (tY - self.lam_b * tX)], 1)
        return torch.stack([ban, pick], 2)

    def reg(self):
        return (self.phi.weight ** 2).sum(), (self.delta.weight ** 2).sum(), (self.gamma ** 2).sum() + (self.beta ** 2).sum()


_SEQ_CACHE = {}


def build_tensors(series, V):
    """把每个系列赛的全部候选流程展开成张量（两种先后手各一份）。流程部分与词表无关，按系列赛集合缓存。"""
    P = torch.tensor([[V.ip(s) for s in r.X] + [V.ip(s) for s in r.Y] for r in series.itertuples()])
    PY = torch.tensor([[V.ipy(s, r.year) for s in r.X] + [V.ipy(s, r.year) for s in r.Y] for r in series.itertuples()])
    Y = torch.tensor(list(series.period))
    key = tuple(series.match_id)
    if key not in _SEQ_CACHE:
        _SEQ_CACHE[key] = seq_tensors(series)
    return {"P": P, "PY": PY, "Y": Y, **_SEQ_CACHE[key]}


def seq_tensors(series):
    sidx, team, act, choice, mask = [], [], [], [], []
    for i, r in enumerate(series.itertuples()):
        steps = FORMATS[r.fmt]
        for seq, avails in enumerate_vetoes(r.fmt, r.pool, r.maps):
            for order in (0, 1):                                     # order=0：X 是先手；1：Y 是先手
                sidx.append(i)
                team.append([a ^ order for a, _ in steps])
                act.append([b for _, b in steps])
                choice.append([MAPS.index(m) for m in seq])
                mk = np.zeros((len(steps), NM), bool)
                for k, av in enumerate(avails):
                    mk[k, [MAPS.index(m) for m in av]] = True
                mask.append(mk)
    return {"sidx": torch.tensor(sidx), "team": torch.tensor(team), "act": torch.tensor(act),
            "choice": torch.tensor(choice), "mask": torch.as_tensor(np.stack(mask)), "n": len(series)}


def series_loglik(model, T):
    tX = model.theta(T["P"][:, :5], T["PY"][:, :5])
    tY = model.theta(T["P"][:, 5:], T["PY"][:, 5:])
    L = model.logits(T["Y"], tX, tY)                                  # [Ns, 2, 2, M]
    l = L[T["sidx"].unsqueeze(1), T["team"], T["act"]]                # [Nseq, steps, M]
    l = l.masked_fill(~T["mask"], -1e9)
    lp = torch.log_softmax(l, -1).gather(-1, T["choice"].unsqueeze(-1)).squeeze(-1).sum(-1) + np.log(0.5)
    mx = torch.full((T["n"],), -1e30).scatter_reduce(0, T["sidx"], lp.detach(), reduce="amax")
    s = torch.zeros(T["n"]).index_add(0, T["sidx"], torch.exp(lp - mx[T["sidx"]]))
    return torch.log(s) + mx                                          # [Ns] 每个系列赛的边际对数似然


def fit(series, V, n_periods, l2_phi, l2_delta, use_players=True, epochs=300, lr=0.02, valid=None, log=print):
    T = build_tensors(series, V)
    Tv = build_tensors(valid, V) if valid is not None and len(valid) else None
    model = VetoModel(len(V.p), len(V.py), n_periods, use_players)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = (1e9, None)
    for ep in range(epochs):
        opt.zero_grad()
        nll = -series_loglik(model, T).mean()
        rp, rd, rg = model.reg()
        loss = nll + (l2_phi * rp + l2_delta * rd + 1e-3 * rg) / T["n"]
        loss.backward()
        opt.step()
        if Tv is not None and (ep + 1) % 10 == 0:
            with torch.no_grad():
                v = -series_loglik(model, Tv).mean().item()
            if v < best[0]:
                best = (v, {k: x.clone() for k, x in model.state_dict().items()}, ep + 1)
    if Tv is not None and best[1] is not None:
        model.load_state_dict(best[1])
        if log:
            log(f"    l2=({l2_phi},{l2_delta}) players={use_players}: valid NLL {best[0]:.4f} @ {best[2]} 轮")
        return model, best[0], best[2]
    return model, None, epochs


# ---------------------------------------------------------------- 模拟
def team_ids(V, team):
    return [V.ip(s) for s, _ in team], [V.ipy(s, y) for s, y in team]


def sample_vetoes(model, V, teamA, teamB, fmt, pool, period, n=2000, first=None, rng=None):
    """teamA/teamB: [(steamid, year)]×5。first: 'A' / 'B' / None(随机)。
    返回 choices [n, steps]（图编号）、first_is_A [n]、decider [n]、每步概率 probs [n, steps]。"""
    rng = rng or np.random.default_rng(0)
    with torch.no_grad():
        pa, pya = team_ids(V, teamA)
        pb, pyb = team_ids(V, teamB)
        tA = model.theta(torch.tensor([pa]), torch.tensor([pya]))
        tB = model.theta(torch.tensor([pb]), torch.tensor([pyb]))
        L = model.logits(torch.tensor([period]), tA, tB)[0].numpy()               # [2(A/B), 2(禁/选), M]
    steps = FORMATS[fmt]
    first_A = rng.random(n) < 0.5 if first is None else np.full(n, first == "A")
    avail = np.zeros((n, NM), bool)
    avail[:, [MAPS.index(m) for m in pool]] = True
    choices = np.zeros((n, len(steps)), int)
    probs = np.zeros((n, len(steps)))
    for k, (actor, act) in enumerate(steps):
        is_A = first_A if actor == 0 else ~first_A                    # 这一步由 A 队行动？
        lg = np.where(is_A[:, None], L[0, act][None], L[1, act][None])
        lg = np.where(avail, lg, -np.inf)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        c = (p.cumsum(1) > rng.random(n)[:, None]).argmax(1)
        choices[:, k] = c
        probs[:, k] = p[np.arange(n), c]
        avail[np.arange(n), c] = False
    decider = avail.argmax(1)
    return choices, first_A, decider, probs


def series_win_prob(order, p_map, fmt):
    """按打图顺序 order（图编号）和每张图 A 的胜率，算 A 赢下系列赛的概率。"""
    need = {"bo1": 1, "bo3": 2, "bo5": 3}[fmt]
    dp = {(0, 0): 1.0}
    win = 0.0
    for m in order:
        nxt = {}
        for (a, b), pr in dp.items():
            for da, q in ((1, p_map[m]), (0, 1 - p_map[m])):
                s = (a + da, b + 1 - da)
                if s[0] == need:
                    win += pr * q
                elif s[1] == need:
                    continue
                else:
                    nxt[s] = nxt.get(s, 0) + pr * q
        dp = nxt
    return win


class Veto:
    def __init__(self, model, V, cfg):
        self.model, self.V, self.cfg = model.eval(), V, cfg
        self.presets = sorted(cfg["presets"], key=lambda p: -p["period"])      # 最新的放最前

    @classmethod
    def load(cls, mdir=None):
        mdir = mdir or MDIR
        cfg = json.load(open(f"{mdir}/config.json"))
        V = Vocab.from_dict(json.load(open(f"{mdir}/vocab.json")))
        m = VetoModel(len(V.p), len(V.py), len(cfg["presets"]))
        m.load_state_dict(torch.load(f"{mdir}/veto_model.pt", map_location="cpu"))
        return cls(m, V, cfg)

    def preset(self, label=None):
        for p in self.presets:
            if p["label"] == label:
                return p
        return self.presets[0]

    def simulate(self, teamA, teamB, fmt="bo3", preset=None, first=None, seed=0, n=4000, map_wp=None):
        """模拟 ban/pick。map_wp：{图: A 胜率}（P4），给了就同时算系列赛胜率。
        返回一次具体流程（随剧本编号 seed 变化）+ 各图被禁/被选/成为决胜图的概率。"""
        pr = self.preset(preset)
        pool, year = pr["pool"], pr["year"]
        rng = np.random.default_rng(seed)
        ch, fa, dec, probs = sample_vetoes(self.model, self.V, teamA, teamB, fmt, pool, pr["period"], n=n, first=first, rng=rng)
        steps = FORMATS[fmt]
        stat = {m: {"A禁": 0.0, "B禁": 0.0, "A选": 0.0, "B选": 0.0, "决胜": 0.0} for m in pool}
        for k, (actor, act) in enumerate(steps):
            for i_m, m in enumerate(MAPS):
                if m not in stat:
                    continue
                hit = ch[:, k] == i_m
                a_side = fa if actor == 0 else ~fa
                key = "禁" if act == 0 else "选"
                stat[m]["A" + key] += (hit & a_side).mean()
                stat[m]["B" + key] += (hit & ~a_side).mean()
        for i_m, m in enumerate(MAPS):
            if m in stat:
                stat[m]["决胜"] = (dec == i_m).mean()
        j = 0                                                          # 用第 0 个样本作为本剧本的具体流程
        flow = []
        for k, (actor, act) in enumerate(steps):
            who = "A" if (actor == 0) == bool(fa[j]) else "B"
            flow.append({"team": who, "action": "ban" if act == 0 else "pick", "map": MAPS[ch[j, k]], "prob": float(probs[j, k])})
        flow.append({"team": None, "action": "decider", "map": MAPS[dec[j]], "prob": float(stat[MAPS[dec[j]]]["决胜"])})
        order = [f["map"] for f in flow if f["action"] in ("pick", "decider")]
        out = {"fmt": fmt, "pool": pool, "year": year, "preset": pr["label"], "first": "A" if fa[j] else "B", "flow": flow,
               "order": order, "stat": stat}
        if map_wp is not None:
            out["series_wp_flow"] = series_win_prob([MAPS.index(m) for m in order], {MAPS.index(k): v for k, v in map_wp.items()}, fmt)
            wp = {MAPS.index(k): v for k, v in map_wp.items()}
            orders = [list(ch[i][[k for k, (_, a) in enumerate(steps) if a == 1]]) + [dec[i]] for i in range(min(n, 1500))]
            out["series_wp"] = float(np.mean([series_win_prob(o, wp, fmt) for o in orders]))
        return out


# ---------------------------------------------------------------- 评估 + 训练
def evaluate(model, V, test, n=3000, seed=0):
    """对每个盲测系列赛记录：边际 NLL；BO3 预测的两张选图命中几张、真实决胜图的预测概率/是否 top1；BO1 同理。"""
    T = build_tensors(test, V)
    with torch.no_grad():
        nll = (-series_loglik(model, T)).numpy()
    rng = np.random.default_rng(seed)
    recs = []
    for i, r in enumerate(test.itertuples()):
        A = [(s, r.year) for s in r.X]
        B = [(s, r.year) for s in r.Y]
        ch, fa, dec, _ = sample_vetoes(model, V, A, B, r.fmt, r.pool, r.period, n=n, rng=rng)
        steps = FORMATS[r.fmt]
        rec = {"match_id": r.match_id, "fmt": r.fmt, "nll": float(nll[i])}
        dfreq = np.bincount(dec, minlength=NM) / n
        if r.fmt == "bo3":
            pk = np.concatenate([ch[:, k] for k, (_, a) in enumerate(steps) if a == 1])
            top2 = set(np.argsort(-np.bincount(pk, minlength=NM))[:2])
            rec["pick_hit"] = len(top2 & {MAPS.index(m) for m in r.maps[:2]}) / 2
            if len(r.maps) == 3:
                rec["dec_p"] = float(dfreq[MAPS.index(r.maps[2])])
                rec["dec_top1"] = float(dfreq.argmax() == MAPS.index(r.maps[2]))
        else:
            rec["dec_p"] = float(dfreq[MAPS.index(r.maps[0])])
            rec["dec_top1"] = float(dfreq.argmax() == MAPS.index(r.maps[0]))
        recs.append(rec)
    return pd.DataFrame(recs)


def sanity_check(model, V, series, period, top=20):
    """对出场最多的 top 个战队-年份：模型下它最常禁的图，是否在它自己最少打的两张图里；最常选的图是否在它最常打的两张图里。
    对手用全体平均（θ=0），只看自身习惯。"""
    rows = []
    cnt = pd.concat([series[["slugX", "year", "X", "pool", "period"]].set_axis(["slug", "year", "roster", "pool", "period"], axis=1),
                     series[["slugY", "year", "Y", "pool", "period"]].set_axis(["slug", "year", "roster", "pool", "period"], axis=1)])
    for (slug, y), g in sorted(cnt.groupby(["slug", "year"]), key=lambda kv: -len(kv[1]))[:top]:
        from collections import Counter
        roster = [s for s, _ in Counter(s for r in g.roster for s in r).most_common(5)]
        pool = g.pool.iloc[-1]
        per = int(g.period.iloc[-1])
        played = maps_played(series, slug, y)
        share = {m: played.get(m, 0) for m in pool}
        low2 = set(sorted(pool, key=lambda m: share[m])[:2])
        high2 = set(sorted(pool, key=lambda m: -share[m])[:2])
        with torch.no_grad():
            p, py = team_ids(V, [(s, y) for s in roster])
            tT = model.theta(torch.tensor([p]), torch.tensor([py]))
            L = model.logits(torch.tensor([per]), tT, torch.zeros_like(tT))[0].numpy()
        idx = [MAPS.index(m) for m in pool]
        ban = pool[int(np.argmax(L[0, 0][idx]))]
        pick = pool[int(np.argmax(L[0, 1][idx]))]
        rows.append({"team": f"{slug} {y}", "最常禁": ban, "在最少打的两张里": ban in low2, "最常选": pick, "在最常打的两张里": pick in high2})
    df = pd.DataFrame(rows)
    return {"最常禁的图 ∈ 自己最少打的两张": round(df["在最少打的两张里"].mean(), 3), "最常选的图 ∈ 自己最常打的两张": round(df["在最常打的两张里"].mean(), 3),
            "战队数": len(df), "明细": rows}


def maps_played(series, slug, y):
    from collections import Counter
    c = Counter()
    for r in series[(series.year == y) & ((series.slugX == slug) | (series.slugY == slug))].itertuples():
        c.update(r.maps)
    return c


def summarize_eval(df):
    b3, b1 = df[df.fmt == "bo3"], df[df.fmt == "bo1"]
    d3 = b3.dropna(subset=["dec_p"]) if "dec_p" in b3 else b3.iloc[:0]
    return {"NLL/系列赛": round(df.nll.mean(), 4), "BO3 两张选图命中率": round(b3.pick_hit.mean(), 3),
            "BO3 决胜图 top1": round(d3.dec_top1.mean(), 3) if len(d3) else None,
            "BO3 真实决胜图的预测概率": round(d3.dec_p.mean(), 3) if len(d3) else None,
            "BO1 比赛图 top1": round(b1.dec_top1.mean(), 3) if len(b1) else None,
            "BO1 真实图的预测概率": round(b1.dec_p.mean(), 3) if len(b1) else None,
            "系列赛": len(df), "BO3": len(b3), "BO3 打满三图": len(d3), "BO1": len(b1)}


def paired_bootstrap(a, b, col, n=2000, seed=0):
    """a、b：同一批系列赛上两个模型的逐系列赛记录。返回 a−b 的均值差和 95% 置信区间。"""
    m = a[["match_id", col]].merge(b[["match_id", col]], on="match_id", suffixes=("_a", "_b")).dropna()
    d = (m[col + "_a"] - m[col + "_b"]).values
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"差值": round(float(d.mean()), 4), "95%CI": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "n": len(d)}


def cross_validate(series, n_periods, l2, k=5, seed=0, log=print):
    """按赛事分组的 k 折交叉验证：每折留出约 1/k 的赛事，另从训练赛事里留 10% 系列赛做早停。"""
    ev = np.array(sorted(series.event_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ev)
    folds = np.array_split(ev, k)
    out = {"players": [], "popularity": [], "uniform": []}
    for i, fe in enumerate(folds):
        te = series[series.event_id.isin(fe)]
        tr_all = series[~series.event_id.isin(fe)]
        va = tr_all.sample(frac=0.1, random_state=i)
        tr = tr_all.drop(va.index)
        V = Vocab(tr)
        mp, _, ep = fit(tr, V, n_periods, *l2, True, 400, valid=va, log=None)
        mb, _, eb = fit(tr, V, n_periods, 0, 0, False, 400, valid=va, log=None)
        for key, m in (("players", mp), ("popularity", mb), ("uniform", VetoModel(1, 1, n_periods, False))):
            out[key].append(evaluate(m, V, te, seed=i))
        log(f"  第 {i + 1}/{k} 折：盲测 {len(te)} 个系列赛（{len(fe)} 个赛事），早停 {ep}/{eb} 轮，"
            f"NLL 选手偏好 {out['players'][-1].nll.mean():.3f} / 只看热度 {out['popularity'][-1].nll.mean():.3f}")
    return {k_: pd.concat(v) for k_, v in out.items()}


def save(model, V, presets, mdir, report):
    os.makedirs(mdir, exist_ok=True)
    torch.save(model.state_dict(), f"{mdir}/veto_model.pt")
    json.dump(V.to_dict(), open(f"{mdir}/vocab.json", "w"))
    json.dump({"presets": presets, "maps": MAPS, "report": report}, open(f"{mdir}/config.json", "w"), ensure_ascii=False, indent=1)


def train_main():
    t0 = time.time()
    maps = pd.read_parquet(f"{C}/maps.parquet")
    maps = maps[~maps.map_uid.isin(p4.excluded_maps())]
    series, presets = load_series(maps)
    NP = len(presets)
    print(f"系列赛 {len(series)}（BO3 {int((series.fmt == 'bo3').sum())}，BO1 {int((series.fmt == 'bo1').sum())}），地图池时期 {NP} 个，{time.time() - t0:.0f}s")
    if not p4.FINAL:
        # 1) 超参数：和 P4 相同的赛事划分（训练 / 验证 E58–E61），按验证集 NLL 选 L2
        valid = series[series.event_id.isin(p4.VALID_EVENTS)]
        train = series[~series.event_id.isin(p4.TEST_EVENTS | p4.VALID_EVENTS)]
        V = Vocab(train)
        grid = [(1.0, 5.0), (3.0, 10.0), (10.0, 30.0)] if not p4.QUICK else [(3.0, 10.0)]
        ep = 30 if p4.QUICK else 400
        res = [fit(train, V, NP, a, b, True, ep, valid=valid) for a, b in grid]
        k = int(np.argmin([r[1] for r in res]))
        best_l2 = grid[k]
        # 2) 时间盲测：在训练+验证上训练，测 E62–E65
        test = series[series.event_id.isin(p4.TEST_EVENTS)]
        trv = pd.concat([train, valid])
        V2 = Vocab(trv)
        va2 = trv.sample(frac=0.1, random_state=0)
        mp, _, ep_p = fit(trv.drop(va2.index), V2, NP, *best_l2, True, ep, valid=va2, log=None)
        mb, _, _ = fit(trv.drop(va2.index), V2, NP, 0, 0, False, ep, valid=va2, log=None)
        tp, tb, tu = (evaluate(m, V2, test) for m in (mp, mb, VetoModel(1, 1, NP, False)))
        report = {"超参数": {"l2": best_l2, "epochs": ep_p},
                  "时间盲测（E62–E65）": {"选手偏好模型": summarize_eval(tp), "只看时期地图热度": summarize_eval(tb), "完全随机": summarize_eval(tu)}}
        # 3) 按赛事分组的 5 折交叉验证 + 配对 bootstrap（样本更多，结论更稳）
        if not p4.QUICK:
            cv = cross_validate(series, NP, best_l2)
            report["5 折交叉验证（按赛事分组）"] = {"选手偏好模型": summarize_eval(cv["players"]), "只看时期地图热度": summarize_eval(cv["popularity"]),
                                         "完全随机": summarize_eval(cv["uniform"])}
            report["配对 bootstrap：选手偏好 − 只看热度"] = {
                "NLL（越低越好）": paired_bootstrap(cv["players"], cv["popularity"], "nll"),
                "BO3 两张选图命中率": paired_bootstrap(cv["players"], cv["popularity"], "pick_hit"),
                "真实决胜/比赛图的预测概率": paired_bootstrap(cv["players"], cv["popularity"], "dec_p"),
                "决胜/比赛图 top1": paired_bootstrap(cv["players"], cv["popularity"], "dec_top1")}
        print(json.dumps(report, ensure_ascii=False, indent=1))
        os.makedirs(p4.MDIR, exist_ok=True)
        json.dump(report, open(os.path.join(p4.MDIR, "veto_eval.json"), "w"), ensure_ascii=False, indent=1)
        best_ep = ep_p
    else:
        prev = os.path.join(os.path.dirname(os.environ.get("CS2_CALIB_SRC", "")), "veto", "config.json")   # 沿用现役模型的超参数
        report = json.load(open(prev))["report"] if os.path.exists(prev) else {}
        hp = report.get("超参数", {})
        best_l2, best_ep = tuple(hp.get("l2", (3.0, 10.0))), int(hp.get("epochs", 100))
    V = Vocab(series)
    model, _, _ = fit(series, V, NP, *best_l2, True, 30 if p4.QUICK else best_ep)
    report["正式模型参数"] = {"lam_p": round(float(model.lam_p), 3), "lam_b": round(float(model.lam_b), 3), "s_b": round(float(model.s_b), 3)}
    report["合理性检查（出场最多的 20 个战队-年份）"] = sanity_check(model, V, series, NP)
    print(json.dumps({k: v for k, v in report["合理性检查（出场最多的 20 个战队-年份）"].items() if k != "明细"}, ensure_ascii=False),
          report["正式模型参数"])
    save(model, V, presets, MDIR, report)
    print(f"正式 Ban/Pick 模型已保存到 {MDIR}（全部 {len(series)} 个系列赛，{best_ep} 轮），总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    else:
        print(__doc__)
