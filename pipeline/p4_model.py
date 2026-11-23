"""P4 第一版对战模拟器：选手-年份向量 + 回合胜负模型 + 经济抽样 + MR12 整场蒙特卡洛。

用法：
    python p4_model.py train      训练回合模型、构建经济抽样池、在盲测赛事上评估，输出 models/p4_report.txt
    python p4_model.py demo       用训练好的模型跑几个示例对决
作为库：
    import p4_model as sim
    S = sim.Simulator.load()
    S.simulate([("76561198...", 2025), ...], [(...), ...], "de_mirage", n=2000)

数据划分（按时间，盲测赛事从头到尾不参与训练和调参）：
    test  = E62–E65（2026 年最后 4 个赛事）
    valid = E58–E61（用于早停）
    train = 其余全部
"""
import paths
import json
import math
import os
import pickle
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

C = paths.CLEAN
MDIR = paths.MODELS
TEST_EVENTS = {"E62", "E63", "E64", "E65"}
VALID_EVENTS = {"E58", "E59", "E60", "E61"}
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train", "de_vertigo"]
MIN_ROUNDS_PLAYER = 50       # 训练集中至少这么多回合才有自己的向量，否则用「替补水平」向量
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FINAL = os.environ.get("CS2_FINAL") == "1"     # 正式模型：用全部数据训练（盲测已在评估阶段完成），轮数固定
QUICK = os.environ.get("CS2_QUICK") == "1"     # 流程测试：极少轮数


# ======================================================================== 数据
def load_rounds():
    maps = pd.read_parquet(f"{C}/maps.parquet")
    rd = pd.read_parquet(f"{C}/rounds.parquet")
    rp = pd.read_parquet(f"{C}/round_players.parquet")
    rd = rd.merge(maps[["map_uid", "match_id", "year", "event_id", "map_name", "legal"]], on="map_uid")
    rd = rd[rd.winner_team.isin([1, 2])].copy()
    rp = rp.sort_values(["map_uid", "real_round", "team", "equip"], ascending=[True, True, True, False])
    rp["slot"] = rp.groupby(["map_uid", "real_round", "team"]).cumcount()
    rp = rp[rp.slot < 5]
    key = ["map_uid", "real_round"]
    sid = rp.pivot_table(index=key, columns=["team", "slot"], values="steamid", aggfunc="first")
    eq = rp.pivot_table(index=key, columns=["team", "slot"], values="equip", aggfunc="first")
    rd = rd.set_index(key).join(sid.set_axis([f"p{t}_{s}" for t, s in sid.columns], axis=1), how="inner")
    rd = rd.join(eq.set_axis([f"e{t}_{s}" for t, s in eq.columns], axis=1), how="inner").reset_index()
    rd["split"] = np.where(rd.event_id.isin(TEST_EVENTS), "test", np.where(rd.event_id.isin(VALID_EVENTS), "valid", "train"))
    ex = excluded_maps()
    if ex:
        rd = rd[~rd.map_uid.isin(ex)].reset_index(drop=True)
    return maps, rd


def excluded_maps():
    """自动更新做把关时，候选模型训练要排除新数据：CS2_EXCLUDE_MAPS 指向一个每行一个 map_uid 的文件。"""
    f = os.environ.get("CS2_EXCLUDE_MAPS")
    return set(open(f).read().split()) if f and os.path.exists(f) else set()


def final_train_split(rd):
    """正式模型：训练集 = 全部回合；验证集只用来打印日志。"""
    return rd.reset_index(drop=True), rd[rd.split == "valid"].reset_index(drop=True)


class Vocab:
    def __init__(self, train_rd):
        cnt_p, cnt_py = defaultdict(int), defaultdict(int)
        for t in (1, 2):
            for s in range(5):
                col = train_rd[f"p{t}_{s}"].dropna()
                yrs = train_rd.loc[col.index, "year"]
                for p, y in zip(col, yrs):
                    cnt_p[p] += 1
                    cnt_py[(p, int(y))] += 1
        self.p = {p: i + 1 for i, p in enumerate(sorted(k for k, v in cnt_p.items() if v >= MIN_ROUNDS_PLAYER))}
        self.py = {k: i + 1 for i, k in enumerate(sorted(k for k, v in cnt_py.items() if v >= MIN_ROUNDS_PLAYER))}
        self.rounds_p, self.rounds_py = dict(cnt_p), {f"{k[0]}|{k[1]}": v for k, v in cnt_py.items()}

    def ip(self, p):
        return self.p.get(p, 0)

    def ipy(self, p, y):
        return self.py.get((p, int(y)), 0)

    def to_json(self):
        return {"p": self.p, "py": {f"{k[0]}|{k[1]}": v for k, v in self.py.items()},
                "rounds_p": self.rounds_p, "rounds_py": self.rounds_py}

    @classmethod
    def from_json(cls, d):
        v = cls.__new__(cls)
        v.p = d["p"]
        v.py = {(k.split("|")[0], int(k.split("|")[1])): i for k, i in d["py"].items()}
        v.rounds_p, v.rounds_py = d["rounds_p"], d["rounds_py"]
        return v


def ctx_features(own, opp, real_round, is_ct):
    """回合情境（从某一方视角）：攻防、半场进度、手枪局、加时、比分。"""
    rr = np.asarray(real_round)
    in_half = np.where(rr <= 24, (rr - 1) % 12, (rr - 25) % 3) / 12.0
    return np.stack([is_ct, in_half, np.isin(rr, [1, 13]).astype(float), (rr > 24).astype(float),
                     (own - opp) / 13.0, own / 13.0, opp / 13.0], axis=1).astype(np.float32)


def encode(rd, vocab):
    n = len(rd)
    P = np.zeros((2, n, 5), np.int64)
    PY = np.zeros((2, n, 5), np.int64)
    E = np.zeros((2, n, 5), np.float32)
    yrs = rd.year.values
    for ti, t in enumerate((1, 2)):
        for s in range(5):
            ps = rd[f"p{t}_{s}"].values
            P[ti, :, s] = [vocab.ip(p) if isinstance(p, str) else 0 for p in ps]
            PY[ti, :, s] = [vocab.ipy(p, y) if isinstance(p, str) else 0 for p, y in zip(ps, yrs)]
            E[ti, :, s] = np.nan_to_num(rd[f"e{t}_{s}"].values.astype(float)) / 1000.0
    side1 = (rd.ct_team.values == 1).astype(np.float32)
    s1, s2 = rd.score1_before.values.astype(float), rd.score2_before.values.astype(float)
    ctx1 = ctx_features(s1, s2, rd.real_round.values, side1)
    ctx2 = ctx_features(s2, s1, rd.real_round.values, 1 - side1)
    m = np.array([MAPS.index(x) for x in rd.map_name.values], np.int64)
    y = (rd.winner_team.values == 1).astype(np.float32)
    return dict(p1=P[0], py1=PY[0], e1=E[0], p2=P[1], py2=PY[1], e2=E[1], side1=side1, ctx1=ctx1, ctx2=ctx2, m=m, y=y)


# ======================================================================== 回合模型
CFG = {"d": 32, "h": 128, "n_models": 5}


class RoundModel(nn.Module):
    def __init__(self, n_p, n_py, d=CFG["d"], h=CFG["h"]):
        super().__init__()
        self.pe = nn.Embedding(n_p, d)
        self.pye = nn.Embedding(n_py, d)
        nn.init.normal_(self.pe.weight, std=0.05)
        nn.init.zeros_(self.pye.weight)
        self.me = nn.Embedding(len(MAPS), 16)
        self.player = nn.Sequential(nn.Linear(d + 2 + 16, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU())
        self.head = nn.Sequential(nn.Linear(4 * h + 7 + 16, h), nn.GELU(), nn.Dropout(0.2), nn.Linear(h, 1))

    def team(self, p, py, e, side, m):
        v = self.pe(p) + self.pye(py)
        x = torch.cat([v, e.unsqueeze(-1), side[:, None, None].expand(-1, 5, 1), m[:, None, :].expand(-1, 5, -1)], -1)
        h = self.player(x)
        return torch.cat([h.mean(1), h.max(1).values], -1)

    def forward(self, p1, py1, e1, p2, py2, e2, side1, ctx1, ctx2, m):
        me = self.me(m)
        A = self.team(p1, py1, e1, side1, me)
        B = self.team(p2, py2, e2, 1 - side1, me)
        sA = self.head(torch.cat([A, B, ctx1, me], -1)).squeeze(-1)
        sB = self.head(torch.cat([B, A, ctx2, me], -1)).squeeze(-1)
        return sA - sB                       # 反对称：交换两队，胜率互补


class Ensemble(nn.Module):
    """多个回合模型取 logit 平均。"""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args):
        return torch.stack([m(*args) for m in self.models]).mean(0)


def batches(D, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield {k: torch.as_tensor(v[j], device=DEV) for k, v in D.items()}


def run_model(model, b):
    return model(b["p1"], b["py1"], b["e1"], b["p2"], b["py2"], b["e2"], b["side1"], b["ctx1"], b["ctx2"], b["m"])


def train_round_model(Dtr, Dva, n_p, n_py, epochs=40, bs=1024, lr=1e-3, seed=0, patience=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RoundModel(n_p, n_py).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best, best_state, bad = 1e9, None, 0
    ntr = len(Dtr["y"])
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(ntr)
        for b in batches(Dtr, perm, bs):
            # 随机把一些选手换成「未知」，学出替补水平；随机去掉年份修正，让通用向量承载主要实力
            for k in ("p1", "p2"):
                drop = torch.rand_like(b[k], dtype=torch.float) < 0.05
                b[k] = torch.where(drop, torch.zeros_like(b[k]), b[k])
                b["py" + k[1]] = torch.where(drop, torch.zeros_like(b["py" + k[1]]), b["py" + k[1]])
            for k in ("py1", "py2"):
                drop = torch.rand_like(b[k], dtype=torch.float) < 0.2
                b[k] = torch.where(drop, torch.zeros_like(b[k]), b[k])
            logit = run_model(model, b)
            loss = F.binary_cross_entropy_with_logits(logit, b["y"]) + 1e-2 * model.pye.weight.pow(2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = evaluate_rounds(model, Dva)
        print(f"  seed {seed} epoch {ep + 1}: valid logloss {va:.4f}", flush=True)
        if va < best - 1e-4:
            best, best_state, bad = va, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if FINAL:                                  # 正式模型不按验证集挑选，用最后一轮
        return model, va
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict_rounds(model, D, bs=8192):
    model.eval()
    out = [torch.sigmoid(run_model(model, b)).cpu().numpy() for b in batches(D, np.arange(len(D["y"])), bs)]
    return np.concatenate(out)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate_rounds(model, D):
    return logloss(predict_rounds(model, D), D["y"])


# ======================================================================== 经济抽样
def econ_state_rows(rd):
    """每回合每队一行：状态 → 该队 5 人装备（降序）。"""
    rows = []
    rd = rd.sort_values(["map_uid", "real_round"])
    for uid, g in rd.groupby("map_uid", sort=False):
        prev_win = {1: None, 2: None}
        streak = {1: 0, 2: 0}
        prev_eq = {1: None, 2: None}
        last_rr = 0
        for r in g.itertuples(index=False):
            if r.real_round != last_rr + 1:            # 有缺失回合时状态不可靠，重置
                prev_win, streak, prev_eq = {1: None, 2: None}, {1: 0, 2: 0}, {1: None, 2: None}
            for t in (1, 2):
                eqs = sorted([getattr(r, f"e{t}_{s}") for s in range(5) if not pd.isna(getattr(r, f"e{t}_{s}"))], reverse=True)
                if len(eqs) == 5:
                    rows.append((econ_key(r.real_round, prev_win[t], streak[t], prev_eq[t]), eqs))
            for t in (1, 2):
                won = r.winner_team == t
                eqs = [getattr(r, f"e{t}_{s}") for s in range(5)]
                prev_eq[t] = float(np.nanmean(eqs))
                prev_win[t] = won
                streak[t] = 0 if won else min(streak[t] + 1, 4)
            last_rr = r.real_round
    return rows


def econ_key(real_round, prev_win, streak, prev_eq):
    if real_round in (1, 13):
        return ("pistol",)
    if real_round > 24:
        return ("ot_first",) if (real_round - 25) % 3 == 0 else ("ot", bool(prev_win))
    if prev_win is None or prev_eq is None:
        return ("reg_unknown",)
    b = 0 if prev_eq < 1500 else (1 if prev_eq < 3500 else 2)
    after_pistol = real_round in (2, 14)
    return ("reg", bool(prev_win), streak, b, after_pistol)


def coarse(key):
    if key[0] == "reg":
        return ("reg", key[1], min(key[2], 2))
    return key[:1]


class EconPool:
    def __init__(self, rows):
        fine, crs = defaultdict(list), defaultdict(list)
        for k, eqs in rows:
            fine[k].append(eqs)
            crs[coarse(k)].append(eqs)
        self.fine = {k: np.array(v, np.float32) for k, v in fine.items() if len(v) >= 30}
        self.coarse = {k: np.array(v, np.float32) for k, v in crs.items()}
        self.all = np.array([e for _, e in rows], np.float32)

    def to_dict(self):
        return {"fine": self.fine, "coarse": self.coarse, "all": self.all}

    @classmethod
    def from_dict(cls, d):
        p = cls.__new__(cls)
        p.fine, p.coarse, p.all = d["fine"], d["coarse"], d["all"]
        return p

    def sample(self, key, n, rng):
        arr = self.fine.get(key)
        if arr is None:
            arr = self.coarse.get(coarse(key), self.all)
        return arr[rng.integers(0, len(arr), n)]


# ======================================================================== 整场模拟
class Simulator:
    def __init__(self, model, vocab, pool, eq_rank, calib=None):
        self.model, self.vocab, self.pool, self.eq_rank = model.eval(), vocab, pool, eq_rank
        self.calib = calib or {"a": 1.0, "b": 0.0}       # 整场胜率校准：sigmoid(a * logit(p) + b)

    @classmethod
    def load(cls, mdir=MDIR):
        vocab = Vocab.from_json(json.load(open(f"{mdir}/vocab.json")))
        cfg = json.load(open(f"{mdir}/config.json"))
        models = []
        for i in range(cfg["n_models"]):
            m = RoundModel(len(vocab.p) + 1, len(vocab.py) + 1, d=cfg["d"], h=cfg["h"]).to(DEV)
            m.load_state_dict(torch.load(f"{mdir}/round_model_{i}.pt", map_location=DEV))
            models.append(m)
        class _U(pickle.Unpickler):            # 兼容以脚本方式运行时保存的旧文件（类记在 __main__ 下）
            def find_class(self, module, name):
                return getattr(sys.modules[__name__], name) if module == "__main__" else super().find_class(module, name)
        obj = _U(open(f"{mdir}/econ_pool.pkl", "rb")).load()
        pool = EconPool.from_dict(obj) if isinstance(obj, dict) else obj
        eq_rank = json.load(open(f"{mdir}/equip_rank.json"))
        calib = json.load(open(f"{mdir}/calib.json")) if os.path.exists(f"{mdir}/calib.json") else None
        return cls(Ensemble(models), vocab, pool, eq_rank, calib)

    def calibrate(self, p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = self.calib["a"] * np.log(p / (1 - p)) + self.calib["b"]
        return float(1 / (1 + np.exp(-z)))

    def _team_tensors(self, team, n):
        """team: [(steamid, year)] × 5 → 按「平时装备高低」排好序的索引（高装备的人分到高装备）。"""
        team = sorted(team, key=lambda x: -self.eq_rank.get(x[0], 0))
        p = torch.tensor([self.vocab.ip(s) for s, _ in team], device=DEV).repeat(n, 1)
        py = torch.tensor([self.vocab.ipy(s, y) for s, y in team], device=DEV).repeat(n, 1)
        return p, py

    @torch.no_grad()
    def simulate(self, teamA, teamB, map_name, n=2000, seed=0, a_starts_ct=None):
        rng = np.random.default_rng(seed)
        pA, pyA = self._team_tensors(teamA, n)
        pB, pyB = self._team_tensors(teamB, n)
        m = torch.full((n,), MAPS.index(map_name), device=DEV)
        a_ct_first = rng.random(n) < 0.5 if a_starts_ct is None else np.full(n, a_starts_ct)
        sa, sb = np.zeros(n, int), np.zeros(n, int)
        st = {t: dict(win=np.array([None] * n, object), streak=np.zeros(n, int), eq=np.full(n, np.nan)) for t in "AB"}
        active = np.ones(n, bool)
        rr = 0
        while active.any() and rr < 60:
            rr += 1
            idx = np.where(active)[0]
            k = len(idx)
            # 第 rr 回合 A 是否守 CT
            if rr <= 12:
                a_ct = a_ct_first[idx]
            elif rr <= 24:
                a_ct = ~a_ct_first[idx]
            else:   # 加时：25–27 沿用下半场阵营，之后每 6 回合换一次（加时开始不换边）
                a_ct = ~a_ct_first[idx] if ((rr - 22) // 6) % 2 == 0 else a_ct_first[idx]
            eqs = {}
            for t in "AB":
                e = np.zeros((k, 5), np.float32)
                keys = [econ_key(rr, st[t]["win"][i], st[t]["streak"][i], st[t]["eq"][i]) for i in idx]
                groups = defaultdict(list)
                for j, key in enumerate(keys):
                    groups[key].append(j)
                for key, js in groups.items():
                    e[js] = self.pool.sample(key, len(js), rng)
                eqs[t] = e
            own, opp = sa[idx].astype(float), sb[idx].astype(float)
            side1 = a_ct.astype(np.float32)
            ctx1 = torch.as_tensor(ctx_features(own, opp, np.full(k, rr), side1), device=DEV)
            ctx2 = torch.as_tensor(ctx_features(opp, own, np.full(k, rr), 1 - side1), device=DEV)
            ii = torch.as_tensor(idx, device=DEV)
            logit = self.model(pA[ii], pyA[ii], torch.as_tensor(eqs["A"] / 1000, device=DEV),
                               pB[ii], pyB[ii], torch.as_tensor(eqs["B"] / 1000, device=DEV),
                               torch.as_tensor(side1, device=DEV), ctx1, ctx2, m[ii])
            pa = torch.sigmoid(logit).cpu().numpy()
            a_won = rng.random(k) < pa
            sa[idx] += a_won
            sb[idx] += ~a_won
            for t, won in (("A", a_won), ("B", ~a_won)):
                st[t]["eq"][idx] = eqs[t].mean(1)
                st[t]["win"][idx] = won
                st[t]["streak"][idx] = np.where(won, 0, np.minimum(st[t]["streak"][idx] + 1, 4))
            done = np.array([match_over(a, b) for a, b in zip(sa[idx], sb[idx])])
            active[idx[done]] = False
        winA = sa > sb
        scores = pd.Series([f"{a}-{b}" for a, b in zip(sa, sb)]).value_counts(normalize=True)
        return {"p_win_A": self.calibrate(winA.mean()), "p_win_A_raw": float(winA.mean()),
                "avg_rounds_A": float(sa.mean()), "avg_rounds_B": float(sb.mean()),
                "p_overtime": float(((sa + sb) > 24).mean()), "top_scores": scores.head(8).round(4).to_dict()}


def match_over(a, b):
    hi, lo = max(a, b), min(a, b)
    if hi == 13 and lo <= 11:
        return True
    if hi < 16:
        return False
    return (hi - 13) % 3 == 0 and hi - lo >= 2 and (hi - lo) <= 4


# ======================================================================== 基线：战队 Elo
def elo_predictions(maps, k=16):
    mp = maps[maps.legal.astype(bool) & maps.team1_slug.notna() & maps.team2_slug.notna()].copy()
    mp["mid"] = pd.to_numeric(mp.match_id, errors="coerce")
    mp = mp.sort_values(["mid", "map_number"])
    R = defaultdict(lambda: 1500.0)
    preds = {}
    for r in mp.itertuples(index=False):
        a, b = r.team1_slug, r.team2_slug
        pa = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))
        preds[r.map_uid] = pa
        if r.event_id in TEST_EVENTS:      # 盲测期间不更新（和模型一样只用赛前信息）
            continue
        s = 1.0 if r.score1 > r.score2 else 0.0
        R[a] += k * (s - pa)
        R[b] -= k * (s - pa)
    return preds


# ======================================================================== 训练 + 评估
def train_main():
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    maps, rd = load_rounds()
    tr, va, te = (rd[rd.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    if FINAL:
        tr, va = final_train_split(rd)
    print(f"回合：train {len(tr)} / valid {len(va)} / test {len(te)}；设备 {DEV}", flush=True)
    vocab = Vocab(tr)
    print(f"有向量的选手 {len(vocab.p)}，选手-年份 {len(vocab.py)}", flush=True)
    Dtr, Dva, Dte = encode(tr, vocab), encode(va, vocab), encode(te, vocab)

    # 基线 1：只看双方平均装备 + 攻防 + 地图（逻辑回归）
    def base_feats(D):
        diff = D["e1"].mean(1) - D["e2"].mean(1)
        side = D["side1"] * 2 - 1
        mo = np.eye(len(MAPS))[D["m"]] * side[:, None]
        return np.column_stack([diff, side, diff * side, D["ctx1"][:, 2] * side, mo])
    Xtr, Xte = base_feats(Dtr), base_feats(Dte)
    w = torch.zeros(Xtr.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)
    Xt, yt = torch.tensor(np.column_stack([Xtr, np.ones(len(Xtr))]), dtype=torch.float32), torch.tensor(Dtr["y"])

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(Xt @ w, yt)
        l.backward()
        return l
    opt.step(closure)
    p_base = torch.sigmoid(torch.tensor(np.column_stack([Xte, np.ones(len(Xte))]), dtype=torch.float32) @ w).detach().numpy()

    models, vas = [], []
    for seed in range(CFG["n_models"]):
        if QUICK and seed >= 2:
            break
        m_, va_ = train_round_model(Dtr, Dva, len(vocab.p) + 1, len(vocab.py) + 1, seed=seed, epochs=1 if QUICK else (3 if FINAL else 40), patience=99 if FINAL else 4)
        models.append(m_.eval())
        vas.append(va_)
        torch.save(m_.state_dict(), f"{MDIR}/round_model_{seed}.pt")
    model = Ensemble(models).eval()
    best_va = evaluate_rounds(model, Dva)
    p_te = predict_rounds(model, Dte)
    json.dump(vocab.to_json(), open(f"{MDIR}/vocab.json", "w"))
    json.dump({**CFG, "n_models": len(models)}, open(f"{MDIR}/config.json", "w"))

    # 经济池（只用训练集）和「平时装备高低」
    pool = EconPool(econ_state_rows(tr))
    pickle.dump(pool.to_dict(), open(f"{MDIR}/econ_pool.pkl", "wb"))
    rp = pd.read_parquet(f"{C}/round_players.parquet", columns=["map_uid", "steamid", "equip"])
    rp = rp[rp.map_uid.isin(set(tr.map_uid))]
    eq_rank = rp.groupby("steamid").equip.mean().round(1).to_dict()
    json.dump(eq_rank, open(f"{MDIR}/equip_rank.json", "w"))

    S = Simulator(model, vocab, pool, eq_rank)
    if FINAL:                                  # 校准参数沿用评估阶段在验证集上拟合的结果
        import shutil
        src = os.environ.get("CS2_CALIB_SRC", os.path.join(paths.MODELS, "calib.json"))
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(f"{MDIR}/calib.json"):
            shutil.copy(src, f"{MDIR}/calib.json")
        print("正式 P4 模型已保存到", MDIR, flush=True)
        return

    def sim_maps(events):
        rows = []
        for r in maps[maps.event_id.isin(events) & maps.legal.astype(bool)].itertuples(index=False):
            A = [(s, int(r.year)) for s in str(r.roster1).split(",") if s]
            B = [(s, int(r.year)) for s in str(r.roster2).split(",") if s]
            if len(A) != 5 or len(B) != 5:
                continue
            out = S.simulate(A, B, r.map_name, n=1000, seed=zlib.crc32(r.map_uid.encode()))
            rows.append({"map_uid": r.map_uid, "event_id": r.event_id, "p_raw": out["p_win_A_raw"], "y": float(r.score1 > r.score2),
                         "pred_rounds_A": out["avg_rounds_A"], "pred_rounds_B": out["avg_rounds_B"], "score1": r.score1, "score2": r.score2})
        return pd.DataFrame(rows)

    # 校准：在验证赛事上拟合 sigmoid(a * logit(p) + b)
    vres = sim_maps(VALID_EVENTS)
    z = torch.tensor(np.log(np.clip(vres.p_raw.values, 1e-4, 1 - 1e-4) / (1 - np.clip(vres.p_raw.values, 1e-4, 1 - 1e-4))), dtype=torch.float32)
    yv = torch.tensor(vres.y.values, dtype=torch.float32)
    ab = torch.tensor([1.0, 0.0], requires_grad=True)
    opt2 = torch.optim.LBFGS([ab], max_iter=100)

    def closure2():
        opt2.zero_grad()
        l = F.binary_cross_entropy_with_logits(ab[0] * z + ab[1], yv)
        l.backward()
        return l
    opt2.step(closure2)
    S.calib = {"a": float(ab[0]), "b": float(ab[1])}
    json.dump(S.calib, open(f"{MDIR}/calib.json", "w"))

    # Elo：K 在验证赛事上选
    best_k, best_l = None, 1e9
    for k in (4, 8, 12, 16, 24, 32):
        e = elo_predictions(maps, k)
        l = logloss(np.array([e.get(u, 0.5) for u in vres.map_uid]), vres.y.values)
        if l < best_l:
            best_k, best_l = k, l
    elo = elo_predictions(maps, best_k)

    res = sim_maps(TEST_EVENTS)
    res["p_model"] = [S.calibrate(p) for p in res.p_raw]
    res["p_elo"] = [elo.get(u, 0.5) for u in res.map_uid]
    res.to_csv(f"{MDIR}/test_predictions.csv", index=False)

    def map_metrics(p, y):
        return {"logloss": round(logloss(p, y), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
                "acc": round(float(np.mean((p > 0.5) == (y > 0.5))), 4)}
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("# P4 模拟器报告（第二版：5 模型集成 + 校准）", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s，设备 {DEV}")
    P(f"回合数 train {len(tr)} / valid {len(va)} / test {len(te)}；有向量的选手 {len(vocab.p)}，选手-年份 {len(vocab.py)}")
    P("\n## 回合级（盲测赛事，越低越好）")
    P(f"  永远 50%      logloss {logloss(np.full(len(te), 0.5), Dte['y']):.4f}")
    P(f"  只看装备基线  logloss {logloss(p_base, Dte['y']):.4f}")
    P(f"  选手向量模型（{CFG['n_models']} 个集成）logloss {logloss(p_te, Dte['y']):.4f}（valid {best_va:.4f}，单模型 valid {np.round(vas, 4).tolist()}）")
    P(f"\n## 整场级（盲测 {len(res)} 张图，赛前只知道双方阵容和地图）")
    P(f"  永远 50%    {map_metrics(np.full(len(res), 0.5), res.y.values)}")
    P(f"  战队 Elo（K={best_k}，验证集选出）{map_metrics(res.p_elo.values, res.y.values)}")
    P(f"  模拟器（未校准）{map_metrics(res.p_raw.values, res.y.values)}")
    P(f"  模拟器（校准后，a={S.calib['a']:.2f} b={S.calib['b']:.2f}）{map_metrics(res.p_model.values, res.y.values)}")
    P(f"  模拟器+Elo 平均 {map_metrics((res.p_model.values + res.p_elo.values) / 2, res.y.values)}")
    bins = pd.cut(res.p_model, [0, .3, .45, .55, .7, 1])
    P("\n校准（模拟器预测胜率分档 → 实际胜率）:", res.groupby(bins, observed=True).y.agg(["mean", "size"]).round(3).to_dict("index"))
    P(f"\n比分预测：预测回合差 vs 实际回合差 相关系数 {np.corrcoef(res.pred_rounds_A - res.pred_rounds_B, res.score1 - res.score2)[0, 1]:.3f}")
    txt = "\n".join(L)
    open(f"{MDIR}/p4_report.txt", "w").write(txt)
    print(txt)


def demo():
    S = Simulator.load()
    pl = pd.read_parquet(f"{C}/players.parquet")
    maps = pd.read_parquet(f"{C}/maps.parquet")
    name = dict(zip(pl.steamid, pl.name))

    def roster(slug, year):
        m = maps[((maps.team1_slug == slug) | (maps.team2_slug == slug)) & (maps.year == year)]
        ros = np.where(m.team1_slug == slug, m.roster1, m.roster2)
        from collections import Counter
        c = Counter(s for r in ros for s in str(r).split(","))
        return [(s, year) for s, _ in c.most_common(5)]
    for (a, ya), (b, yb), mp in [(("vitality", 2025), ("spirit", 2026), "de_mirage"), (("natus-vincere", 2024), ("vitality", 2025), "de_inferno"),
                                 (("faze", 2023), ("mouz", 2026), "de_nuke")]:
        A, B = roster(a, ya), roster(b, yb)
        if len(A) < 5 or len(B) < 5:
            print("阵容不足:", a, ya, b, yb)
            continue
        out = S.simulate(A, B, mp)
        print(f"{a}-{ya} [{', '.join(name.get(s, s) for s, _ in A)}] vs {b}-{yb} [{', '.join(name.get(s, s) for s, _ in B)}] @ {mp}: "
              f"{a} 胜率 {out['p_win_A']:.1%}，平均比分 {out['avg_rounds_A']:.1f}-{out['avg_rounds_B']:.1f}，最常见 {list(out['top_scores'].items())[:3]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    train_main() if cmd == "train" else demo()
