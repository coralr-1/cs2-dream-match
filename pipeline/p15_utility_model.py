"""V8 道具生成模型（深度学习）：学习每名选手 / 每支队在什么局面下、什么时候、扔什么道具、扔到哪里。

数据：clean/utility.parquet（p14_utility.py 从原始录像提取，约 160 万颗道具）。
落点词表：把真实落点按（地图, 道具类型）切成 96 单位的格子，出现 ≥15 次的格子作为一个「道具点位」（如 Mirage 的 A 包点烟、拱门烟），
          少见的落点并到最近的点位。生成时先选点位，再从这个点位的真实投掷记录里取一条（落点、出手位置、飞行时间、持续时间），
          所以坐标都来自真实比赛。
模型：Transformer。输入 = 回合 token（地图、比分、阵营）+ 两队战队 token + 10 名选手 token（选手 + 年份 + 阵营 + 装备）
      + 这回合剧本的事件 token（击杀 / 下包 / 结束，来自事件模型）+ 已经扔出的道具 token；
      逐颗预测：类型（或结束）→ 出手秒数 → 投掷者（指向 10 名选手之一）→ 道具点位。
生成约束：投掷者必须活着；每人最多 1 烟、1 火、2 闪、1 雷，合计不超过 4 颗；出手时间不早于上一颗。

用法：python p15_utility_model.py train      （GPU 上训练 + 盲测评估，保存到 $CS2_MODELS/utility）
"""
import paths
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import p4_model as p4
import p7_event_model as em

C = em.C
MDIR = paths.MODELS + "/utility"
DEV = em.DEV
MAPS = em.MAPS
UTYPES = ["smoke", "molotov", "flash", "he"]            # 类型编号 1–4；0 = 结束
MAXU = 40                                              # 每回合最多 40 颗
NTU = 160                                              # 出手时间：1 秒一档，0–159 秒
CELL_U = 96.0
MIN_CELL = 15
N_SAMPLES = 80
LIMIT = {"smoke": 1, "molotov": 1, "flash": 2, "he": 1}
CFG = {"d": 256, "layers": 6, "heads": 8}


# ======================================================================== 落点词表
def build_lineups(u, maps_of):
    """返回 (每颗道具的点位编号, 点位表 DataFrame, 每个点位的真实样本 dict)。点位编号从 1 开始，0 = 无。"""
    u = u.assign(map_name=u.map_uid.map(maps_of))
    u["cx"] = np.round(u.x / CELL_U).astype(int)
    u["cy"] = np.round(u.y / CELL_U).astype(int)
    cnt = u.groupby(["map_name", "type", "cx", "cy"]).size().rename("n").reset_index()
    keep = cnt[cnt.n >= MIN_CELL].reset_index(drop=True)
    keep["lid"] = np.arange(1, len(keep) + 1)
    u = u.merge(keep[["map_name", "type", "cx", "cy", "lid"]], on=["map_name", "type", "cx", "cy"], how="left")
    from scipy.spatial import cKDTree
    for (m, t), g in keep.groupby(["map_name", "type"]):                 # 少见落点并到同图同类型最近的点位
        miss = u.lid.isna() & (u.map_name == m) & (u.type == t)
        if miss.any():
            tree = cKDTree(g[["cx", "cy"]].values * CELL_U)
            _, k = tree.query(u.loc[miss, ["x", "y"]].values)
            u.loc[miss, "lid"] = g.lid.values[k]
    u = u.dropna(subset=["lid"])
    u["lid"] = u.lid.astype(int)
    lin = u.groupby("lid").agg(map_name=("map_name", "first"), type=("type", "first"), x=("x", "mean"), y=("y", "mean"), n=("x", "size")).reset_index()
    rng = np.random.default_rng(0)
    samples = {}
    for lid, g in u.groupby("lid"):
        g = g.iloc[rng.permutation(len(g))[:N_SAMPLES]]
        samples[int(lid)] = np.stack([g.x, g.y, g.z, g.x0, g.y0, (g.t_start - g.t_throw).clip(0, 8), (g.t_end - g.t_start).clip(0, 25)], 1).astype(np.float32)
    return u, lin, samples


# ======================================================================== 数组
def utility_arrays(rd, u):
    """每回合的道具序列（按出手时间排序）。输入位置 j=0 是开始符，位置 j 预测第 j+1 颗（或结束）。"""
    R = len(rd)
    key = dict(zip(zip(rd.map_uid, rd.real_round), range(R)))
    u = u.assign(ri=[key.get(k, -1) for k in zip(u.map_uid, u.real_round)])
    u = u[(u.ri >= 0) & (u.slot >= 0)].sort_values(["ri", "t_throw"])
    u["j"] = u.groupby("ri").cumcount()
    u = u[u.j < MAXU]
    U = {k: np.zeros((R, MAXU + 1), np.int64) for k in ("type_in", "t_in", "slot_in", "lid_in", "type_y", "t_y", "slot_y", "lid_y")}
    U["slot_in"][:] = 10
    U["type_in"][:, 0] = 5                                                 # 开始符
    ri, j = u.ri.values, u.j.values
    ty = u.type.map({t: i + 1 for i, t in enumerate(UTYPES)}).values
    tb = np.clip(u.t_throw.values.astype(int), 0, NTU - 1)
    U["type_y"][ri, j], U["t_y"][ri, j], U["slot_y"][ri, j], U["lid_y"][ri, j] = ty, tb, u.slot.values, u.lid.values
    U["type_in"][ri, j + 1], U["t_in"][ri, j + 1], U["slot_in"][ri, j + 1], U["lid_in"][ri, j + 1] = ty, tb, u.slot.values, u.lid.values
    n = np.bincount(ri, minlength=R)
    U["valid"] = (np.arange(MAXU + 1)[None] <= np.minimum(n, MAXU)[:, None]).astype(np.float32)    # 包括最后的「结束」
    U["n"] = n
    return U


# ======================================================================== 模型
class UtilityModel(nn.Module):
    def __init__(self, n_p, n_py, n_place, n_weapon, n_team, n_team_year, n_lid, d=CFG["d"], layers=CFG["layers"], heads=CFG["heads"]):
        super().__init__()
        self.d = d
        E = nn.Embedding
        self.pe, self.pye = E(n_p, d), E(n_py, d)
        nn.init.normal_(self.pe.weight, std=0.02)
        nn.init.zeros_(self.pye.weight)
        self.tme, self.tmye = E(n_team, d), E(n_team_year, d)
        nn.init.normal_(self.tme.weight, std=0.02)
        nn.init.zeros_(self.tmye.weight)
        self.team_e, self.side_e, self.slot_e, self.absent_e = E(2, d), E(2, d), E(11, d), E(2, d)
        self.eq_proj = nn.Linear(1, d)
        self.map_e, self.round_proj = E(len(MAPS), d), nn.Linear(7, d)
        # 剧本事件（与事件模型同样的编码）
        self.etype_e, self.et_e, self.w_e, self.hs_e = E(5, d), E(em.NT, d), E(n_weapon, d), E(2, d)
        self.apl_e, self.vpl_e, self.site_e, self.reason_e, self.epos_e = E(n_place, d), E(n_place, d), E(3, d), E(6, d), E(em.MAXEV, d)
        self.vslot_e = E(11, d)
        # 道具 token
        self.utype_e, self.ut_e, self.lid_e, self.upos_e = E(6, d), E(NTU, d), E(n_lid, d), E(MAXU + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.nctx = 13
        L = self.nctx + em.MAXEV + MAXU + 1
        m = torch.zeros(L, L, dtype=torch.bool)
        ce = self.nctx + em.MAXEV
        m[:ce, ce:] = True                                                   # 上下文 / 剧本看不到道具
        m[ce:, ce:] = torch.triu(torch.ones(MAXU + 1, MAXU + 1, dtype=torch.bool), 1)     # 道具只看已经扔出的
        self.register_buffer("attn_mask", m, persistent=False)
        self.register_buffer("team_ids", torch.tensor([0] * 5 + [1] * 5), persistent=False)
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))
        self.type_head = nn.Linear(d, 5)
        self.time_head = mlp(2 * d, NTU)
        self.qs, self.ks = mlp(3 * d, d), nn.Linear(d, d)
        self.lid_head = mlp(4 * d, n_lid)

    def encode(self, b):
        B = b["m"].shape[0]
        dev = b["m"].device
        slot_ids = torch.arange(10, device=dev)
        rt = self.map_e(b["m"]) + self.round_proj(b["rctx"])
        pres = b["present"]
        pt = (self.pe(b["p"]) + self.pye(b["py"]) + self.team_e(self.team_ids)[None] + self.side_e(b["side"]) +
              self.slot_e(slot_ids)[None] + self.eq_proj(b["eq"].unsqueeze(-1))) * pres.unsqueeze(-1) + self.absent_e((pres == 0).long())
        two = torch.arange(2, device=dev)
        tt = self.tme(b["tm"]) + self.tmye(b["tmy"]) + self.team_e(two)[None] + self.side_e(b["side"][:, [0, 5]])
        ev = (self.etype_e(b["type"]) + self.et_e(b["t"]) + self.slot_e(b["a"]) + self.vslot_e(b["v"]) + self.w_e(b["w"]) + self.hs_e(b["hs"]) +
              self.apl_e(b["apl"]) + self.vpl_e(b["vpl"]) + self.site_e(b["site"]) + self.reason_e(b["reason"]) +
              self.epos_e(torch.arange(em.MAXEV, device=dev))[None])
        ut = (self.utype_e(b["type_in"]) + self.ut_e(b["t_in"]) + self.slot_e(b["slot_in"]) + self.lid_e(b["lid_in"]) +
              self.upos_e(torch.arange(MAXU + 1, device=dev))[None])
        x = torch.cat([rt[:, None], tt, pt, ev, ut], 1)
        pad = torch.cat([torch.zeros(B, self.nctx, dtype=torch.bool, device=dev), b["type"] == 0, b["type_in"] == 0], 1)
        h = self.norm(self.tf(x, mask=self.attn_mask, src_key_padding_mask=pad))
        return h[:, 3:13], h[:, self.nctx + em.MAXEV:]                       # 选手 token、道具位置的隐状态

    def heads(self, S, H, type_y, t_y, slot_y):
        """训练时按真实的 类型 → 时间 → 投掷者 逐级条件化（teacher forcing）。"""
        te = self.utype_e(type_y)
        tm = self.ut_e(t_y)
        Ss = torch.gather(S, 1, slot_y.clamp(0, 9).unsqueeze(-1).expand(-1, -1, S.shape[-1]))
        return {"type": self.type_head(H), "t": self.time_head(torch.cat([H, te], -1)),
                "slot": torch.einsum("bld,bsd->bls", self.qs(torch.cat([H, te, tm], -1)), self.ks(S)) / math.sqrt(self.d),
                "lid": self.lid_head(torch.cat([H, te, tm, Ss], -1))}


def to_dev(A, U, idx):
    f32 = lambda a: a.astype(np.float32) if a.dtype == np.float64 else a
    b = {k: torch.as_tensor(f32(v[idx]), device=DEV) for k, v in A.items() if k not in ("win1",)}
    b.update({k: torch.as_tensor(v[idx], device=DEV) for k, v in U.items() if k != "n"})
    return b


def compute_loss(model, b, lid_mask):
    S, H = model.encode(b)
    o = model.heads(S, H, b["type_y"], b["t_y"], b["slot_y"])
    v = b["valid"]
    is_u = (b["type_y"] > 0).float() * v
    ce = lambda lg, y, m: (F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(m.shape) * m).sum() / m.sum().clamp(min=1)
    l_type = ce(o["type"], b["type_y"], v)
    l_t = ce(o["t"], b["t_y"], is_u)
    slot_logit = o["slot"].masked_fill((b["present"] == 0).unsqueeze(1), -1e9)
    l_slot = ce(slot_logit, b["slot_y"].clamp(0, 9), is_u)
    lm = lid_mask[b["m"].unsqueeze(1).expand_as(b["type_y"]), b["type_y"]]   # [B, L, n_lid] 当前地图 + 类型允许的点位
    lid_logit = o["lid"].masked_fill(~lm, -1e9)
    l_lid = ce(lid_logit, b["lid_y"], is_u)
    acc = ((lid_logit.argmax(-1) == b["lid_y"]).float() * is_u).sum() / is_u.sum().clamp(min=1)
    return l_type + l_t + l_slot + l_lid, {"type": float(l_type), "t": float(l_t), "slot": float(l_slot), "lid": float(l_lid), "lid_acc": float(acc)}


# ======================================================================== 生成
class UtilityGen:
    def __init__(self, model, V, lin, samples, lid_mask):
        self.model, self.V, self.lin, self.samples, self.lid_mask = model.eval(), V, lin, samples, lid_mask

    @classmethod
    def load(cls, mdir=None):
        mdir = mdir or MDIR
        V = em.Vocab.from_json(json.load(open(f"{mdir}/vocab.json")))
        cfg = json.load(open(f"{mdir}/config.json"))
        lin = pd.read_parquet(f"{mdir}/lineups.parquet")
        S = dict(np.load(f"{mdir}/lineup_samples.npz"))
        samples = {int(k[1:]): v for k, v in S.items()}
        model = UtilityModel(len(V.p) + 1, len(V.py) + 1, len(V.places) + 1, len(V.weapons) + 2, len(V.team) + 1, len(V.team_year) + 1,
                             cfg["n_lid"]).to(DEV)
        model.load_state_dict(torch.load(f"{mdir}/utility_model.pt", map_location=DEV))
        return cls(model, V, lin, samples, build_lid_mask(lin, cfg["n_lid"]).to(DEV))

    def round_batch(self, story, r, team_ids):
        """把剧本里的一个回合转成模型输入（batch=1）。"""
        V = self.V
        roster = [(p["steamid"], p["year"]) for p in story["players"]]
        a_ct = r["a_side"] == "CT"
        side = np.array([int(a_ct)] * 5 + [int(not a_ct)] * 5)
        sa, sb = r["score_before"]
        b = {"m": [MAPS.index(story["map"])], "rctx": p4.ctx_features(np.array([sa], float), np.array([sb], float), np.array([r["round"]]), np.array([float(a_ct)], np.float32)),
             "p": [[V.ip(s) for s, _ in roster]], "py": [[V.ipy(s, y) for s, y in roster]], "side": [side], "eq": [np.array(r["equip"]) / 1000.0],
             "present": [np.ones(10)], "tm": [[V.itm(t[0]) if t else 0 for t in team_ids]], "tmy": [[V.itmy(t[0], t[1]) if t else 0 for t in team_ids]]}
        E = {k: np.zeros((1, em.MAXEV), np.int64) for k in ("type", "t", "a", "v", "w", "hs", "apl", "vpl", "site", "reason")}
        E["type"][0, 0] = 1
        E["a"][:], E["v"][:] = 10, 10
        for j, e in enumerate(r["events"][:em.MAXEV - 1], start=1):
            E["t"][0, j] = min(max(int(e["time"] / em.TBIN), 0), em.NT - 1)
            if e["type"] == "kill":
                E["type"][0, j], E["a"][0, j], E["v"][0, j] = 2, e["attacker"], e["victim"]
                E["w"][0, j], E["hs"][0, j] = V.iw(e.get("weapon")), int(e.get("headshot", False))
                E["apl"][0, j], E["vpl"][0, j] = V.ipl(e.get("attacker_place")), V.ipl(e.get("victim_place"))
            elif e["type"] == "plant":
                E["type"][0, j], E["a"][0, j], E["site"][0, j] = 3, e["player"], 1 if e["site"] == "A" else 2
            else:
                E["type"][0, j] = 4
                E["reason"][0, j] = em.REASONS.index(e["reason"]) + 1 if e.get("reason") in em.REASONS else 0
        out = {}
        for k, v in {**b, **E}.items():
            a = np.asarray(v)
            out[k] = torch.as_tensor(a.astype(np.float32) if a.dtype == np.float64 else a, device=DEV)
        return out

    @torch.no_grad()
    def generate(self, story, r, team_ids, seed=0, temperature=1.0):
        """返回这一回合的道具列表：[{type, slot, t_throw, t_start, t_end, x, y, z, x0, y0}]。"""
        rng = np.random.default_rng(seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
        b = self.round_batch(story, r, team_ids)
        death = {e["victim"]: e["time"] for e in r["events"] if e["type"] == "kill"}
        end_t = max([e["time"] for e in r["events"] if e["type"] == "end"] or [115.0])
        used = {s: {t: 0 for t in UTYPES} for s in range(10)}
        seq = {k: torch.zeros(1, MAXU + 1, dtype=torch.long, device=DEV) for k in ("type_in", "t_in", "slot_in", "lid_in")}
        seq["slot_in"][:] = 10
        seq["type_in"][0, 0] = 5
        out, t_prev = [], 0
        mi = MAPS.index(story["map"])
        for j in range(MAXU):
            S, H = self.model.encode({**b, **seq})
            h = H[:, j:j + 1]
            # 类型（含结束）：用完额度的类型不能再选
            lt = self.model.type_head(h)[0, 0] / temperature
            for ti, t in enumerate(UTYPES, start=1):
                if all(used[s][t] >= LIMIT[t] or sum(used[s].values()) >= 4 for s in range(10)):
                    lt[ti] = -1e9
            ty = int(torch.multinomial(torch.softmax(lt.float().cpu(), -1), 1, generator=g))
            if ty == 0:
                break
            tname = UTYPES[ty - 1]
            te = self.model.utype_e(torch.tensor([[ty]], device=DEV))
            # 出手时间：不早于上一颗、不晚于回合结束
            ltm = self.model.time_head(torch.cat([h, te], -1))[0, 0].float().cpu() / temperature
            bins = torch.arange(NTU)
            ltm[(bins < t_prev) | (bins > end_t)] = -1e9
            tb = int(torch.multinomial(torch.softmax(ltm, -1), 1, generator=g))
            # 投掷者：活着、还有这种道具
            tm = self.model.ut_e(torch.tensor([[tb]], device=DEV))
            ls = (torch.einsum("bld,bsd->bls", self.model.qs(torch.cat([h, te, tm], -1)), self.model.ks(S)) / math.sqrt(self.model.d))[0, 0].float().cpu()
            for s in range(10):
                if death.get(s, 1e9) <= tb or used[s][tname] >= LIMIT[tname] or sum(used[s].values()) >= 4:
                    ls[s] = -1e9
            if (ls < -1e8).all():
                break
            sl = int(torch.multinomial(torch.softmax(ls / temperature, -1), 1, generator=g))
            # 道具点位：当前地图 + 类型
            Ss = S[:, sl:sl + 1]
            ll = self.model.lid_head(torch.cat([h, te, tm, Ss], -1))[0, 0].float()
            ll = ll.masked_fill(~self.lid_mask[mi, ty], -1e9).cpu() / temperature
            lid = int(torch.multinomial(torch.softmax(ll, -1), 1, generator=g))
            smp = self.samples.get(lid)
            if smp is None or not len(smp):
                continue
            x, y, z, x0, y0, flight, dur = smp[rng.integers(len(smp))]
            t_throw = tb + float(rng.uniform(0, 1))
            out.append({"type": tname, "slot": sl, "t_throw": round(t_throw, 2), "t_start": round(t_throw + float(flight), 2),
                        "t_end": round(t_throw + float(flight) + float(dur), 2), "x": float(x), "y": float(y), "z": float(z), "x0": float(x0), "y0": float(y0),
                        "lineup": lid})
            used[sl][tname] += 1
            t_prev = tb
            for k, v in (("type_in", ty), ("t_in", tb), ("slot_in", sl), ("lid_in", lid)):
                seq[k][0, j + 1] = v
        return out


def build_lid_mask(lin, n_lid):
    m = torch.zeros(len(MAPS), 5, n_lid, dtype=torch.bool)
    for r in lin.itertuples():
        m[MAPS.index(r.map_name), UTYPES.index(r.type) + 1, int(r.lid)] = True
    return m


# ======================================================================== 训练 + 评估
def train_main(epochs=None, bs=256, lr=3e-4):
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    maps, rd, sl, ev = em.load_tables()
    u = pd.read_parquet(f"{C}/utility.parquet")
    u = u[u.map_uid.isin(set(rd.map_uid))]
    maps_of = dict(zip(maps.map_uid, maps.map_name))
    u, lin, samples = build_lineups(u, maps_of)
    n_lid = int(lin.lid.max()) + 1
    print(f"道具 {len(u)} 颗，点位 {len(lin)} 个（{lin.groupby('type').size().to_dict()}），{time.time() - t0:.0f}s", flush=True)
    V = em.Vocab(rd, sl, ev)
    A = em.build_arrays(rd, sl, ev, V)
    U = utility_arrays(rd, u)
    lid_mask = build_lid_mask(lin, n_lid).to(DEV)
    final = p4.FINAL
    tr = np.where(rd.split.values != "test")[0] if final else np.where(rd.split.values == "train")[0]
    va = np.where(rd.split.values == "valid")[0]
    te = np.where(rd.split.values == "test")[0]
    model = UtilityModel(len(V.p) + 1, len(V.py) + 1, len(V.places) + 1, len(V.weapons) + 2, len(V.team) + 1, len(V.team_year) + 1, n_lid).to(DEV)
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M；训练 {len(tr)} / 验证 {len(va)} / 盲测 {len(te)} 回合", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    epochs = epochs or (2 if p4.QUICK else int(os.environ.get("CS2_UTIL_EPOCHS", 14)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(tr) // bs + 1), pct_start=0.1)
    rng = np.random.default_rng(0)
    use_amp = DEV == "cuda"

    def evaluate(idx):
        model.eval()
        tot, n = {}, 0
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                b = to_dev(A, U, idx[s:s + 1024])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, parts = compute_loss(model, b, lid_mask)
                k = len(idx[s:s + 1024])
                for kk, vv in parts.items():
                    tot[kk] = tot.get(kk, 0) + vv * k
                n += k
        model.train()
        return {k: round(v / n, 4) for k, v in tot.items()}

    best = (1e9, None)
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            b = to_dev(A, U, perm[s:s + bs])
            for kk in ("tm", "tmy", "py"):                                    # 随机丢掉战队 / 年份，学到通用水平
                drop = torch.rand(b[kk].shape, device=DEV) < 0.2
                b[kk] = torch.where(drop, torch.zeros_like(b[kk]), b[kk])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = compute_loss(model, b, lid_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vm = evaluate(va)
        vl = vm["type"] + vm["t"] + vm["slot"] + vm["lid"]
        print(f"  第 {ep + 1}/{epochs} 轮 验证 {vm}  {time.time() - t0:.0f}s", flush=True)
        if final or vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save(model.state_dict(), f"{MDIR}/utility_model.pt")
    json.dump(V.to_json(), open(f"{MDIR}/vocab.json", "w"))
    lin.to_parquet(f"{MDIR}/lineups.parquet", index=False)
    np.savez_compressed(f"{MDIR}/lineup_samples.npz", **{f"l{k}": v for k, v in samples.items()})
    report = {"n_lid": n_lid, "epochs": epochs, "final": final}
    if not final:
        report["盲测（逐项对数损失，越低越好）"] = evaluate(te)
        report.update(eval_generation(model, V, lin, samples, lid_mask, rd, sl, ev, u, te, A))
    json.dump(report, open(f"{MDIR}/config.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"道具模型已保存到 {MDIR}，总用时 {time.time() - t0:.0f}s")


def baseline_nll(u, rd, te_idx):
    """频率基线：点位按（地图, 类型）的整体频率。"""
    rt = rd.iloc[te_idx][["map_uid", "real_round"]]
    tr = u.merge(rd[rd.split == "train"][["map_uid", "real_round"]], on=["map_uid", "real_round"])
    tst = u.merge(rt, on=["map_uid", "real_round"])
    freq = tr.groupby(["map_name", "type", "lid"]).size()
    tot = tr.groupby(["map_name", "type"]).size()
    p = [(freq.get((m, t, l), 0) + 0.5) / (tot.get((m, t), 0) + 0.5 * 50) for m, t, l in zip(tst.map_name, tst.type, tst.lid)]
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def eval_generation(model, V, lin, samples, lid_mask, rd, sl, ev, u, te, A, n_rounds=600):
    """用盲测回合的真实剧本（事件、阵容、装备）生成道具，和真实道具比较：每回合各类数量、出手时间、点位分布、每名选手的使用量。"""
    G = UtilityGen(model, V, lin, samples, lid_mask)
    rng = np.random.default_rng(1)
    pick = rng.choice(te, min(n_rounds, len(te)), replace=False)
    gen_rows, real_rows = [], []
    slots = sl.set_index(["map_uid", "real_round"])
    evs = ev.set_index(["map_uid", "real_round"]).sort_index()
    ub = u.set_index(["map_uid", "real_round"]).sort_index()
    for ri in pick:
        r0 = rd.iloc[ri]
        key = (r0.map_uid, r0.real_round)
        try:
            s = slots.loc[key].sort_values("slot")
        except KeyError:
            continue
        if len(s) != 10:
            continue
        players = [{"steamid": str(x), "year": int(r0.year)} for x in s.steamid]
        e = evs.loc[[key]] if key in evs.index else evs.iloc[:0]
        events = []
        for x in e.itertuples():
            if x.type == "kill" and x.a_slot >= 0 and x.v_slot >= 0:
                events.append({"type": "kill", "time": float(x.time), "attacker": int(x.a_slot), "victim": int(x.v_slot), "weapon": x.weapon,
                               "headshot": bool(x.hs), "attacker_place": x.a_place, "victim_place": x.v_place})
            elif x.type == "plant" and x.a_slot >= 0:
                events.append({"type": "plant", "time": float(x.time), "player": int(x.a_slot), "site": x.site})
            elif x.type == "end":
                events.append({"type": "end", "time": float(x.time), "reason": x.reason})
        a_ct = bool(s.side.iloc[0] == "ct")
        story = {"map": r0.map_name, "players": players}
        r = {"a_side": "CT" if a_ct else "T", "score_before": (int(r0.score1_before), int(r0.score2_before)), "round": int(r0.real_round),
             "equip": list(s.equip.astype(float)), "events": events}
        gu = G.generate(story, r, (None, None) if not V.itm(r0.team1_slug) else ((r0.team1_slug, int(r0.year)), (r0.team2_slug, int(r0.year))), seed=int(ri))
        for x in gu:
            gen_rows.append({"ri": ri, "map_name": r0.map_name, "type": x["type"], "t": x["t_throw"], "lid": x["lineup"], "steamid": players[x["slot"]]["steamid"]})
        if key in ub.index:
            for x in ub.loc[[key]].itertuples():
                if x.slot >= 0:
                    real_rows.append({"ri": ri, "map_name": r0.map_name, "type": x.type, "t": x.t_throw, "lid": x.lid, "steamid": players[x.slot]["steamid"]})
    g, rr = pd.DataFrame(gen_rows), pd.DataFrame(real_rows)
    nr = len(pick)
    res = {"生成评估回合数": int(nr)}
    res["每回合道具数（生成 / 真实）"] = {t: [round((g.type == t).sum() / nr, 2), round((rr.type == t).sum() / nr, 2)] for t in UTYPES}
    res["出手时间中位数（生成 / 真实）"] = {t: [round(float(g[g.type == t].t.median()), 1), round(float(rr[rr.type == t].t.median()), 1)] for t in UTYPES}
    js = []
    for (m, t), gg in rr.groupby(["map_name", "type"]):
        pg = g[(g.map_name == m) & (g.type == t)].lid.value_counts(normalize=True)
        pr = gg.lid.value_counts(normalize=True)
        idx = pg.index.union(pr.index)
        a, b2 = pg.reindex(idx, fill_value=0).values, pr.reindex(idx, fill_value=0).values
        mm = 0.5 * (a + b2)
        kl = lambda x, y: np.sum(np.where(x > 0, x * np.log(x / np.maximum(y, 1e-12)), 0))
        js.append((0.5 * kl(a, mm) + 0.5 * kl(b2, mm), len(gg)))
    res["点位分布 JS 散度（按地图×类型加权，0 = 完全一致）"] = round(float(sum(j * n for j, n in js) / sum(n for _, n in js)), 4)
    pc_g = g.groupby(["steamid", "type"]).size().unstack(fill_value=0)
    pc_r = rr.groupby(["steamid", "type"]).size().unstack(fill_value=0)
    both = pc_g.index.intersection(pc_r.index)
    res["每名选手各类道具使用量的相关系数（生成 vs 真实）"] = {t: round(float(np.corrcoef(pc_g.loc[both, t], pc_r.loc[both, t])[0, 1]), 3)
                                              for t in UTYPES if t in pc_g and t in pc_r}
    res["点位频率基线的点位对数损失"] = round(baseline_nll(u, rd, te), 4)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    else:
        print(__doc__)
