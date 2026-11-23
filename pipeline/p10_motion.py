"""V3b 走位生成模型：以「事件剧本」和双方选手/战队习惯为条件，逐秒生成 10 名选手的位置与朝向。

用法：
    python p10_motion.py train      训练，输出 <模型目录>/motion/（模型、可走区域、出生点、地图范围、报告）
作为库：
    import p10_motion as mo
    M = mo.Motion.load()
    tracks = M.generate(cond, plan, n_steps, seed)      # [n_steps, 10, 4]：x, y, yaw, alive

结构（类似 Scene Transformer）：
    记忆 token：回合 token + 2 个战队 token + 10 个选手 token + 最多 14 个事件剧本 token（时间、攻击者/受害者、点位、坐标、包点）
    走位 token：每秒每人一个（位置的傅里叶特征 + 朝向 + 存活 + 回合时钟 + 位置号 + 选手/战队/阵营向量）
    自注意力：只能看当前及更早的秒（同一秒内 10 人互相可见），窗口 32 秒；交叉注意力看记忆
    输出：下一秒的位移（21×21 格，每格 26 单位）和朝向（36 档）
    生成时用真实站位密度得到的「可走区域」防止穿墙，用剧本中的死亡时间决定何时倒下
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
MDIR = paths.MODELS + "/motion"
DEV = em.DEV
MAPS = em.MAPS
W = 32                 # 自注意力窗口（秒）
K = 160
GRID = 21
CELL = 26.0
NYAW = 36
NF = 6                 # 傅里叶频率数
WALK_CELL = 32.0
CFG = {"d": 256, "layers": 6, "heads": 8}


def fourier(u):
    """u: [..., 2] 归一化坐标（约 -1~1）→ [..., 4*NF]"""
    f = (2.0 ** torch.arange(NF, device=u.device, dtype=u.dtype)) * math.pi
    a = u.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


# ======================================================================== 数据
def load_data():
    maps, rd, sl, ev = em.load_tables()
    V = em.Vocab(rd, sl, ev)
    A = em.build_arrays(rd, sl, ev, V)
    bounds = json.load(open(f"{C}/map_bounds.json"))
    B = np.array([bounds.get(m, [-3000, 3000, -3000, 3000]) for m in MAPS], np.float32)
    # 剧本中击杀/下包坐标（与 build_arrays 的事件位置对齐）
    rdi = rd.assign(ri=np.arange(len(rd)))
    e = ev.merge(rdi[["map_uid", "real_round", "ri"]], on=["map_uid", "real_round"]).sort_values(["ri", "seq"])
    e["j"] = e.groupby("ri").cumcount() + 1
    e = e[e.j < em.MAXEV]
    mi = np.array([MAPS.index(m) for m in rd.map_name])[e.ri.values]
    def norm(x, y):
        bx = B[mi]
        return np.stack([(x - bx[:, 0]) / (bx[:, 1] - bx[:, 0]) * 2 - 1, (y - bx[:, 2]) / (bx[:, 3] - bx[:, 2]) * 2 - 1], -1)
    A["vxy"] = np.zeros((len(rd), em.MAXEV, 2), np.float32)
    A["axy"] = np.zeros((len(rd), em.MAXEV, 2), np.float32)
    A["vxy"][e.ri.values, e.j.values] = np.nan_to_num(norm(e.vx.values.astype(float), e.vy.values.astype(float)))
    A["axy"][e.ri.values, e.j.values] = np.nan_to_num(norm(e.ax.values.astype(float), e.ay.values.astype(float)))
    A["t1"] = np.clip((A["t"] * em.TBIN).astype(np.int64), 0, K - 1)        # 剧本时间换成 1 秒粒度
    # 走位数组对齐到回合
    ix = pd.read_parquet(f"{C}/tracks_index.parquet")
    row = rd[["map_uid", "real_round"]].merge(ix, on=["map_uid", "real_round"], how="left")
    A["trow"] = row.row.fillna(-1).astype(np.int64).values
    A["nsteps"] = row.n_steps.fillna(0).astype(np.int64).values
    tracks = np.load(f"{C}/tracks.npy", mmap_mode="r")
    return maps, rd, V, A, B, tracks


# ======================================================================== 模型
class MotionModel(nn.Module):
    def __init__(self, n_p, n_py, n_team, n_tmy, n_place, d=CFG["d"], layers=CFG["layers"], heads=CFG["heads"]):
        super().__init__()
        E = nn.Embedding
        self.d = d
        self.pe, self.pye, self.tme, self.tmye = E(n_p, d), E(n_py, d), E(n_team, d), E(n_tmy, d)
        for e in (self.pe, self.tme):
            nn.init.normal_(e.weight, std=0.02)
        for e in (self.pye, self.tmye):
            nn.init.zeros_(e.weight)
        self.team_e, self.side_e, self.slot_e, self.map_e = E(2, d), E(2, d), E(11, d), E(len(MAPS), d)
        self.round_proj, self.eq_proj = nn.Linear(7, d), nn.Linear(1, d)
        self.type_e, self.time_e, self.v_e, self.a_e, self.pl_e, self.site_e = E(5, d), E(K, d), E(11, d), E(11, d), E(n_place, d), E(3, d)
        self.xy_proj = nn.Linear(8 * NF, d)
        self.pos_proj = nn.Linear(4 * NF + 2, d)
        self.alive_e = E(2, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.move_head = nn.Linear(d, GRID * GRID)
        self.yaw_head = nn.Linear(d, NYAW)
        kk = torch.arange(W).repeat_interleave(10)
        self.register_buffer("tmask", kk.view(1, -1) > kk.view(-1, 1), persistent=False)   # 只能看当前及更早的秒
        self.register_buffer("team_ids", torch.tensor([0] * 5 + [1] * 5), persistent=False)

    def memory(self, b):
        rt = self.map_e(b["m"]) + self.round_proj(b["rctx"])
        two = torch.arange(2, device=b["m"].device)
        tt = self.tme(b["tm"]) + self.tmye(b["tmy"]) + self.team_e(two)[None] + self.side_e(b["side"][:, [0, 5]])
        slots = torch.arange(10, device=b["m"].device)
        self_id = self.pe(b["p"]) + self.pye(b["py"])
        pt = self_id + self.team_e(self.team_ids)[None] + self.side_e(b["side"]) + self.slot_e(slots)[None] + self.eq_proj(b["eq"].unsqueeze(-1))
        plan = (self.type_e(b["type"]) + self.time_e(b["t1"]) + self.v_e(b["v"]) + self.a_e(b["a"]) + self.pl_e(b["vpl"]) + self.site_e(b["site"]) +
                self.xy_proj(torch.cat([fourier(b["vxy"]), fourier(b["axy"])], -1)))
        mem = torch.cat([rt[:, None], tt, pt, plan], 1)
        pad = torch.cat([torch.zeros(len(rt), 13, dtype=torch.bool, device=rt.device), b["type"] == 0], 1)
        return mem, pad, self_id

    def forward(self, b, seq):
        """seq: xy [B,L,10,2]（归一化）、yaw [B,L,10]（度）、alive [B,L,10]、k [B,L]（回合秒数）、valid [B,L]"""
        Bn, L = seq["xy"].shape[:2]
        mem, mpad, self_id = self.memory(b)
        yaw = torch.deg2rad(seq["yaw"]).unsqueeze(-1)
        x = self.pos_proj(torch.cat([fourier(seq["xy"]), yaw.sin(), yaw.cos()], -1))
        x = x + self.alive_e(seq["alive"].long()) + self.time_e(seq["k"].clamp(0, K - 1)).unsqueeze(2)
        x = x + (self_id + self.team_e(self.team_ids)[None] + self.side_e(b["side"]) + self.slot_e(torch.arange(10, device=x.device))[None]).unsqueeze(1)
        x = x.reshape(Bn, L * 10, self.d)
        kpad = (~seq["valid"]).repeat_interleave(10, dim=1)
        h = self.norm(self.dec(x, mem, tgt_mask=self.tmask[:L * 10, :L * 10], tgt_key_padding_mask=kpad, memory_key_padding_mask=mpad))
        h = h.view(Bn, L, 10, self.d)
        return self.move_head(h), self.yaw_head(h)


def cond_batch(A, idx):
    keys = ("m", "rctx", "tm", "tmy", "p", "py", "side", "eq", "type", "t1", "v", "a", "vpl", "site", "vxy", "axy")
    return {k: torch.as_tensor(A[k][idx], device=DEV) for k in keys}


def window_batch(T, Bnd, A, idx, rng, train=True):
    """从每个回合取一个 W 秒窗口。T: GPU 上的走位张量 [N,160,10,4]。"""
    rows = torch.as_tensor(A["trow"][idx], device=DEV)
    ns = A["nsteps"][idx]
    s = np.array([rng.integers(0, max(1, n - 2)) if train else max(0, (n - W) // 2) for n in ns])
    k = torch.as_tensor(s, device=DEV).unsqueeze(1) + torch.arange(W + 1, device=DEV).unsqueeze(0)       # 多取 1 秒作为目标
    kc = k.clamp(max=K - 1)
    tr = T[rows.unsqueeze(1), kc].float()                                                                   # [B, W+1, 10, 4]
    valid = (k < torch.as_tensor(ns, device=DEV).unsqueeze(1)) & (tr[..., 0, 3] >= 0)
    bx = Bnd[torch.as_tensor(A["m"][idx], device=DEV)]
    xy = torch.stack([(tr[..., 0] - bx[:, None, None, 0]) / (bx[:, None, None, 1] - bx[:, None, None, 0]) * 2 - 1,
                      (tr[..., 1] - bx[:, None, None, 2]) / (bx[:, None, None, 3] - bx[:, None, None, 2]) * 2 - 1], -1)
    alive = tr[..., 3] > 0
    seq = {"xy": xy[:, :W], "yaw": tr[:, :W, :, 2], "alive": alive[:, :W], "k": kc[:, :W], "valid": valid[:, :W]}
    d = tr[:, 1:, :, :2] - tr[:, :-1, :, :2]
    bxi = (d[..., 0] / CELL).round().clamp(-(GRID // 2), GRID // 2) + GRID // 2
    byi = (d[..., 1] / CELL).round().clamp(-(GRID // 2), GRID // 2) + GRID // 2
    move = (bxi * GRID + byi).long()
    yawt = (((tr[:, 1:, :, 2] % 360) + 360) % 360 / (360 / NYAW)).long().clamp(0, NYAW - 1)
    m = alive[:, :W] & alive[:, 1:] & valid[:, :W].unsqueeze(-1) & valid[:, 1:].unsqueeze(-1)
    return seq, move, yawt, m


def loss_fn(model, T, Bnd, A, idx, rng, train=True):
    b = cond_batch(A, idx)
    if train:
        for k in ("tm", "tmy", "py"):
            drop = torch.rand(b[k].shape, device=DEV) < 0.2
            b[k] = torch.where(drop, torch.zeros_like(b[k]), b[k])
    seq, move, yawt, m = window_batch(T, Bnd, A, idx, rng, train)
    lm, ly = model(b, seq)
    mf = m.float()
    l1 = (F.cross_entropy(lm.reshape(-1, GRID * GRID), move.reshape(-1), reduction="none").view(m.shape) * mf).sum() / mf.sum().clamp(min=1)
    l2 = (F.cross_entropy(ly.reshape(-1, NYAW), yawt.reshape(-1), reduction="none").view(m.shape) * mf).sum() / mf.sum().clamp(min=1)
    acc = ((lm.argmax(-1) == move).float() * mf).sum() / mf.sum().clamp(min=1)
    return l1 + 0.5 * l2, float(l1), float(l2), float(acc)


# ======================================================================== 生成
class Motion:
    def __init__(self, model, V, bounds, walk, spawns):
        self.model, self.V, self.bounds, self.walk, self.spawns = model.eval(), V, bounds, walk, spawns

    @classmethod
    def load(cls, mdir=MDIR):
        V = em.Vocab.from_json(json.load(open(f"{mdir}/vocab.json")))
        model = MotionModel(len(V.p) + 1, len(V.py) + 1, len(V.team) + 1, len(V.team_year) + 1, len(V.places) + 1).to(DEV)
        model.load_state_dict(torch.load(f"{mdir}/motion_model.pt", map_location=DEV))
        walk = dict(np.load(f"{mdir}/walkable.npz"))
        return cls(model, V, np.array(json.load(open(f"{mdir}/bounds.json")), np.float32), walk, json.load(open(f"{mdir}/spawns.json")))

    def norm(self, m, x, y):
        b = self.bounds[m]
        return (x - b[0]) / (b[1] - b[0]) * 2 - 1, (y - b[2]) / (b[3] - b[2]) * 2 - 1

    def walkable(self, map_name, x, y):
        if map_name not in self.walk:                 # 训练集里没有这张图（如 2026 下半年才加入的 Cache）：不做穿墙限制
            return np.ones(len(x), bool)
        g = self.walk[map_name]
        b = self.bounds[MAPS.index(map_name)]
        i = np.clip(((x - b[0]) / WALK_CELL).astype(int), 0, g.shape[0] - 1)
        j = np.clip(((y - b[2]) / WALK_CELL).astype(int), 0, g.shape[1] - 1)
        return g[i, j]

    @torch.no_grad()
    def generate(self, cond, map_name, n_steps, death_k, seed=0, init=None, temperature=1.0, events=None, phys=None, stats=None, utility=None):
        """cond: 单个回合的条件张量字典（batch=1）；death_k: [10] 每个位置号在第几秒死亡（不死为 None）。
        events: 剧本事件（含击杀时间、双方、计划位置、武器）；phys: p13 地图物理。给了 phys 时加上物理约束：
          - 每秒移动的整条路径都不能穿墙（不只是终点）；
          - 击杀前（最多 12 秒，按绕墙路程和剩余时间决定何时出发）把双方沿可走区域引向剧本里的真实对枪位置；
          - 开枪瞬间（击杀所在的前后两秒）双方之间必须有视线（道具/炸弹击杀除外），攻击者朝向受害者；
          - 没有交火安排的敌我双方不能「互相看见」（有视线且在对方 90° 视野里），看见了就该开枪。
        utility: 道具模型生成的道具（可选）：投掷者提前走到真实出手位置；烟雾挡视线；不站在燃烧区域里。
        stats: 可选 dict，累计约束触发/失败次数。
        返回 [n_steps, 10, 4]：x, y, yaw, alive。"""
        import p13_physics as ph
        rng = np.random.default_rng(seed)
        mi = MAPS.index(map_name)
        side = cond["side"][0].cpu().numpy()
        use_phys = phys is not None and phys.has(map_name)
        st_ = stats if stats is not None else {}
        for key in ("los_fix", "los_fix_real", "los_dash", "los_fail", "contact_fix_move", "contact_fix_joint", "contact_fix_yaw", "contact_fail",
                    "replan", "replan_real"):
            st_.setdefault(key, 0)
        out = np.zeros((n_steps, 10, 4), np.float32)
        if init is None:
            sp = self.spawns[map_name][rng.integers(len(self.spawns[map_name]))]
            ct, t = list(sp["ct"]), list(sp["t"])
            rng.shuffle(ct)
            rng.shuffle(t)
            for s in range(10):
                x, y, yw = (ct if side[s] == 1 else t).pop()
                out[0, s, :3] = (x, y, yw)
        else:
            out[0, :, :3] = init
        out[:, :, 3] = 1
        for s in range(10):
            if death_k[s] is not None:
                out[min(death_k[s], n_steps - 1):, s, 3] = 0
        # ---- 物理约束的计划
        D = np.stack([(np.arange(GRID * GRID) // GRID - GRID // 2) * CELL, (np.arange(GRID * GRID) % GRID - GRID // 2) * CELL], -1)
        kills = []
        for e in (events or []):
            if e.get("type") == "kill" and e.get("time", -1) >= 0:
                kills.append((float(e["time"]), int(e["attacker"]), int(e["victim"]), e.get("attacker_xy"), e.get("victim_xy"),
                              ph.needs_los(e.get("weapon", ""))))
        kl = [(t, a, v) for t, a, v, *_ in kills]
        duel = phys.duel if use_phys else {}
        WIN = float(duel.get("接触后开枪窗口（秒）", ph.CONTACT_WIN))           # 看见之后多久必然开枪（真实数据统计）
        P_VFACE = float(duel.get("受害者朝向攻击者", 0.8))                      # 击杀时受害者朝向攻击者的比例（真实数据统计）
        need_los, face = {}, {}
        goals = {s: [] for s in range(10)}                                     # 每人按时间排好的对枪目标 (时间, x, y)
        for t, a, v, axy, vxy, need in kills:
            ks = int(t)
            for slot, tgt in ((a, axy), (v, vxy)):
                if tgt is not None and use_phys:
                    goals[slot].append((t, float(tgt[0]), float(tgt[1])))
            if need:
                for stp in (ks, ks + 1):
                    if 1 <= stp < n_steps:
                        need_los.setdefault(stp, []).append((a, v))
                for stp in (ks - 1, ks, ks + 1):
                    if 1 <= stp < n_steps:
                        face.setdefault(stp, []).append((a, v))
                if 1 <= ks < n_steps and rng.random() < P_VFACE:              # 真实数据里 82% 的受害者被击杀时也朝着攻击者
                    face.setdefault(ks, []).append((v, a))

        REAL_D = {"sniper": 890.0, "rifle": 615.0, "smg": 410.0, "pistol": 515.0, "shotgun": 250.0, "knife": 40.0, "taser": 120.0}
        replan_at = {}                                                        # 开枪前检查能否按时到位，赶不到就临场改约
        for t, a, v, axy, vxy, need in kills:
            if need and use_phys:                                             # 开枪前 6 秒和 3 秒各检查一次
                for back in (6, 3):
                    if int(t) - back >= 1 or back == 3:
                        replan_at.setdefault(max(1, int(t) - back), []).append((t, a, v))
        wcls = {(float(e["time"]), int(e["attacker"]), int(e["victim"])): ph.weapon_class(e.get("weapon", ""))
                for e in (events or []) if e.get("type") == "kill"}

        def replan(t, a, v, k):
            """当前在第 k 秒。若 a 或 v 按跑动速度赶不到计划位置，就在两人各自能跑到的范围里重新挑一对有视线的位置。"""
            reach = 230.0 * max(1.0, t - k) + 40.0
            ok = True
            for slot in (a, v):
                g = next(((tt, gx, gy) for tt, gx, gy in goals[slot] if tt == t), None)
                if g is None:
                    ok = False
                    continue
                f_ = phys.geo_field(map_name, g[1], g[2])
                if float(phys.geo_lookup(map_name, f_, out[k, slot:slot + 1, 0], out[k, slot:slot + 1, 1])[0]) > reach:
                    ok = False
            if ok:
                return
            # 从真实击杀里检索离两人当前位置最近的枪线（同类武器优先），挑两人都能按时跑到的那条
            pa, pv = phys.real_lines_near(map_name, out[k, a, :2], out[k, v, :2], k=400, weapon=wcls.get((t, a, v)))
            if len(pa) == 0:
                return
            ga = phys.geo_lookup(map_name, phys.geo_field(map_name, out[k, a, 0], out[k, a, 1]), pa[:, 0], pa[:, 1])
            gv = phys.geo_lookup(map_name, phys.geo_field(map_name, out[k, v, 0], out[k, v, 1]), pv[:, 0], pv[:, 1])
            over = np.maximum(ga, gv) / reach
            ok = over <= 1.0
            b = int(np.argmin(np.where(ok, ga + gv + rng.uniform(0, 60, len(ga)), np.inf))) if ok.any() else int(np.argmin(over))
            for slot, p in ((a, pa[b]), (v, pv[b])):
                goals[slot] = [(tt, gx, gy) for tt, gx, gy in goals[slot] if tt != t] + [(t, float(p[0]), float(p[1]))]
            st_["replan"] += 1
            st_["replan_real"] += int(ok.any())

        partner = {}                                                          # (击杀时间, 位置号) → 对枪的另一方
        for t, a, v, axy, vxy, need in kills:
            if need:
                partner[(t, a)] = v
                partner[(t, v)] = a
        prepeek_cache = {}

        def prepeek(gx, gy, px, py):
            """对枪位置旁边、对手（在 px,py）看不到的掩体位置：真实打法是先躲在掩体后，开枪前再拉出来。"""
            key = (round(gx), round(gy), round(px), round(py))
            if key not in prepeek_cache:
                f_ = phys.geo_field(map_name, gx, gy)
                ij = np.argwhere(f_ <= 350.0)
                res = None
                if len(ij):
                    o = phys.meta["origin"][map_name]
                    pts = np.stack([o[0] + (ij[:, 0] + 0.5) * ph.CELL, o[1] + (ij[:, 1] + 0.5) * ph.CELL], -1)
                    hidden = ~phys.los_many(map_name, pts[:, 0], pts[:, 1], px, py) & (np.hypot(pts[:, 0] - px, pts[:, 1] - py) >= ph.R_AWARE)
                    if hidden.any():
                        gd = np.where(hidden, f_[ij[:, 0], ij[:, 1]], np.inf)
                        res = tuple(pts[int(np.argmin(gd))])
                prepeek_cache[key] = res
            return prepeek_cache[key]

        def toward(p_from, p_to):
            return float(np.degrees(np.arctan2(p_to[1] - p_from[1], p_to[0] - p_from[0])))

        util = [u for u in (utility or []) if use_phys]
        smokes_at = lambda t: [(u["x"], u["y"]) for u in util if u["type"] == "smoke" and u["t_start"] <= t <= u["t_end"]]
        fires_at = lambda t: [(u["x"], u["y"]) for u in util if u["type"] == "molotov" and u["t_start"] <= t <= u["t_end"]]
        throw_goal = {}                                                     # slot → [(出手时间, x0, y0)]
        for u in util:
            if u.get("x0") is not None and np.isfinite(u["x0"]):
                throw_goal.setdefault(u["slot"], []).append((u["t_throw"], u["x0"], u["y0"]))

        mem_cond = cond
        for k in range(n_steps - 1):
            lo = max(0, k + 1 - W)
            w = out[lo:k + 1]
            nx, ny = self.norm(mi, w[..., 0], w[..., 1])
            seq = {"xy": torch.as_tensor(np.stack([nx, ny], -1)[None], device=DEV), "yaw": torch.as_tensor(w[None, ..., 2], device=DEV),
                   "alive": torch.as_tensor(w[None, ..., 3] > 0, device=DEV), "k": torch.arange(lo, k + 1, device=DEV)[None],
                   "valid": torch.ones(1, k + 1 - lo, dtype=torch.bool, device=DEV)}
            for t, a, v in replan_at.get(k, []):
                replan(t, a, v, k)
            lm, ly = self.model(mem_cond, seq)
            logp = torch.log_softmax(lm[0, -1].float() / temperature, -1).cpu().numpy()        # [10, 441]
            pyw = torch.softmax(ly[0, -1].float() / temperature, -1)
            cur = out[k, :, :2]
            stp = k + 1
            approach = {}                                                     # 正在赶往对枪位置的人：slot → (路程场, 当前路程)
            for slot in range(10):                                            # 沿可走区域引向下一个对枪位置
                nxt_goal = next(((t, gx, gy) for t, gx, gy in sorted(goals[slot]) if t >= stp - 1), None)
                if nxt_goal is None or out[k, slot, 3] <= 0:
                    continue
                t, gx, gy = nxt_goal
                tau = t - stp
                if tau > 20:
                    continue
                q = partner.get((t, slot))
                if q is not None and tau > WIN + 0.5:                              # 还早：先去旁边的掩体位置等
                    pg = next(((tt, px, py) for tt, px, py in goals[q] if tt == t), None)
                    if pg is not None:
                        hp = prepeek(gx, gy, pg[1], pg[2])
                        if hp is not None:
                            gx, gy = hp
                field = phys.geo_field(map_name, gx, gy)
                g_cur = float(phys.geo_lookup(map_name, field, cur[slot:slot + 1, 0], cur[slot:slot + 1, 1])[0])
                if not np.isfinite(g_cur):
                    continue
                need_t = g_cur / 230.0                                            # 跑到位需要的秒数（跑动约 230 单位/秒）
                gw = 1.0 if tau <= need_t + 1.5 else max(0.15, 1.0 - (tau - need_t - 1.5) / 5.0)
                if tau <= need_t + 2.0:
                    approach[slot] = (field, g_cur, tau - need_t)                   # 第三项：时间余量（秒）
                g_c = phys.geo_lookup(map_name, field, cur[slot, 0] + D[:, 0], cur[slot, 1] + D[:, 1])
                beta = 6.0 if tau <= need_t + 1.5 else 3.0                        # 越紧急拉得越强
                logp[slot] += np.where(np.isfinite(g_c), -beta * gw * (g_c - g_cur) / 100.0, -1e4)
            for slot, lst in throw_goal.items():                              # 扔道具的人提前走到这个道具点位的真实出手位置
                if out[k, slot, 3] <= 0 or slot in approach:
                    continue
                tg = next(((t, gx, gy) for t, gx, gy in sorted(lst) if 0 <= t - stp <= 4), None)
                if tg is None or any(0 <= gt - stp <= 6 for gt, _, _ in goals[slot]):
                    continue
                field = phys.geo_field(map_name, tg[1], tg[2])
                g_cur = float(phys.geo_lookup(map_name, field, cur[slot:slot + 1, 0], cur[slot:slot + 1, 1])[0])
                if not np.isfinite(g_cur) or g_cur > 230.0 * 5:
                    continue
                g_c = phys.geo_lookup(map_name, field, cur[slot, 0] + D[:, 0], cur[slot, 1] + D[:, 1])
                logp[slot] += np.where(np.isfinite(g_c), -2.0 * (g_c - g_cur) / 100.0, -1e4)
            fz = fires_at(stp)
            if fz:                                                            # 不往燃烧区域里走
                for slot in range(10):
                    cx_, cy_ = cur[slot, 0] + D[:, 0], cur[slot, 1] + D[:, 1]
                    infire = np.zeros(len(D), bool)
                    for fx, fy in fz:
                        infire |= np.hypot(cx_ - fx, cy_ - fy) < ph.FIRE_R
                    logp[slot] += np.where(infire, -4.0, 0.0)
            P = np.exp(logp - logp.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            moving = (out[k, :, 3] > 0) | (out[k + 1, :, 3] > 0)
            nxt = cur.copy()
            if use_phys:
                # 一次算出每人 441 种走法里哪些整条路径都不穿墙，只在这些里面抽（避免在拐角反复撞墙后原地不动）
                cand_all = cur[:, None, :] + D[None]
                WM = phys.path_walkable(map_name, np.repeat(cur[:, 0:1], len(D), 1), np.repeat(cur[:, 1:2], len(D), 1),
                                        cand_all[..., 0], cand_all[..., 1])
                WM[:, GRID * GRID // 2] = True                                    # 原地不动总是允许的
                P = P * WM
                P /= P.sum(1, keepdims=True)
                for i in np.where(moving)[0]:
                    c = rng.choice(GRID * GRID, p=P[i])
                    p_new = cur[i] + D[c] + rng.uniform(-CELL / 4, CELL / 4, 2)
                    if not phys.path_walkable(map_name, cur[i, 0], cur[i, 1], p_new[0], p_new[1]):
                        p_new = cur[i] + D[c]
                    nxt[i] = p_new
            else:
                todo = moving.copy()
                for _ in range(6):                                        # 落在墙里的重抽，最多 6 次，仍不行就原地不动
                    if not todo.any():
                        break
                    c = np.array([rng.choice(GRID * GRID, p=P[i]) for i in range(10)])
                    cand = cur + D[c] + rng.uniform(-CELL / 2, CELL / 2, (10, 2))
                    ok = self.walkable(map_name, cand[:, 0], cand[:, 1]) & todo
                    nxt[ok] = cand[ok]
                    todo &= ~ok
            out[k + 1, :, :2] = np.where(moving[:, None], nxt, cur)
            yb = torch.multinomial(pyw, 1).squeeze(-1).cpu().numpy()
            out[k + 1, :, 2] = np.where(moving, (yb + 0.5) * (360 / NYAW), out[k, :, 2])
            if not use_phys:
                continue
            new = out[stp]

            def candidates(p):
                cand = cur[p] + D
                return cand, phys.path_walkable(map_name, np.full(len(D), cur[p, 0]), np.full(len(D), cur[p, 1]), cand[:, 0], cand[:, 1])

            def pick(p, cand, mask):
                q = P[p] * mask
                if q.sum() <= 0:
                    q = mask.astype(float)
                return cand[rng.choice(len(cand), p=q / q.sum())]

            # 1) 开枪瞬间要有视线
            for a, v in need_los.get(stp, []):
                if phys.los(map_name, new[a, 0], new[a, 1], new[v, 0], new[v, 1])[0]:
                    continue
                done = False
                for p, q in ((a, v), (v, a)):                                  # 真实枪线：另一端离对方当前位置 ≤ 120，这一端一步可达
                    if not moving[p]:
                        continue
                    la, lv = phys.real_lines_near(map_name, v_xy=new[q, :2] if p == a else None, a_xy=new[q, :2] if p == v else None, k=400)
                    if len(la) == 0:
                        continue
                    mine, other = (la, lv) if p == a else (lv, la)
                    near_q = np.hypot(*(other - new[q, :2]).T) <= 120
                    step_ok = phys.path_walkable(map_name, np.full(len(mine), cur[p, 0]), np.full(len(mine), cur[p, 1]), mine[:, 0], mine[:, 1]) & \
                        (np.hypot(*(mine - cur[p]).T) <= CELL * (GRID // 2) * 1.2)
                    okm = near_q & step_ok
                    if okm.any():
                        okm &= phys.los_many(map_name, mine[:, 0], mine[:, 1], new[q, 0], new[q, 1])
                    if okm.any():
                        cands = np.where(okm)[0]
                        new[p, :2] = mine[cands[np.argmin(np.hypot(*(mine[cands] - new[p, :2]).T))]]
                        st_["los_fix_real"] += 1
                        done = True
                        break
                if not done:                                                     # 最后的保险：沿可走区域快速拉到一条真实枪线上（最多约 480 单位）
                    for p, q in ((a, v), (v, a)):
                        if not moving[p]:
                            continue
                        f_ = phys.geo_field(map_name, cur[p, 0], cur[p, 1])
                        la, lv = phys.real_lines_near(map_name, v_xy=new[q, :2] if p == a else None, a_xy=new[q, :2] if p == v else None, k=600)
                        mine, other = (la, lv) if p == a else (lv, la)
                        if len(mine) == 0:
                            continue
                        pts = mine[(np.hypot(*(other - new[q, :2]).T) <= 160) & (phys.geo_lookup(map_name, f_, mine[:, 0], mine[:, 1]) <= 480.0)]
                        if len(pts) == 0:
                            continue
                        vis = phys.los_many(map_name, pts[:, 0], pts[:, 1], new[q, 0], new[q, 1])
                        if vis.any():
                            gd = np.where(vis, phys.geo_lookup(map_name, f_, pts[:, 0], pts[:, 1]), np.inf)
                            new[p, :2] = pts[int(np.argmin(gd))]
                            st_["los_dash"] += 1
                            done = True
                            break
                for p, q in ((a, v), (v, a)) if not done else ():               # 实在找不到真实枪线：退回到一步可达的几何视线位置
                    if not moving[p]:
                        continue
                    cand, mask = candidates(p)
                    mask &= phys.los_many(map_name, cand[:, 0], cand[:, 1], new[q, 0], new[q, 1])
                    if mask.any():
                        new[p, :2] = pick(p, cand, mask)
                        st_["los_fix"] += 1
                        done = True
                        break
                st_["los_fail"] += int(not done)
                if not done:
                    gp = {}
                    for slot in (a, v):
                        tg = next(((t, gx, gy) for t, gx, gy in sorted(goals[slot]) if t >= stp - 1), None)
                        if tg is not None:
                            f_ = phys.geo_field(map_name, tg[1], tg[2])
                            gp[slot] = float(phys.geo_lookup(map_name, f_, new[slot:slot + 1, 0], new[slot:slot + 1, 1])[0])
                    st_.setdefault("fail_detail", []).append((stp, gp.get(a), gp.get(v), float(np.hypot(*(new[a, :2] - new[v, :2])))))
            # 2) 攻击者开枪时朝向受害者（以及部分受害者朝向攻击者）
            for p, q in face.get(stp, []):
                new[p, 2] = toward(new[p, :2], new[q, :2])
            # 3) 没有交火安排的「互相看见」：先挪位置，挪不开再转开视角
            protect = {}
            for a, v in need_los.get(stp, []):
                protect.setdefault(a, []).append(v)
                protect.setdefault(v, []).append(a)
            al = new[:, 3] > 0

            def bad_pairs():
                """整秒处（新位置 + 新朝向）和视频里的半秒处（插值位置 + 上一秒朝向）都检查。"""
                mid = 0.5 * (cur + new[:, :2])
                found = set(ph.contacts(phys, map_name, new[:, :2], new[:, 2], al, smokes_at(stp))) | \
                    set(ph.contacts(phys, map_name, mid, out[k, :, 2], al, smokes_at(stp - 0.5)))
                return [(i, j) for i, j in sorted(found) if not ph.pair_allowed(kl, stp, i, j, WIN)]
            for _ in range(5):
                bad = bad_pairs()
                if not bad:
                    break
                i, j = bad[0]
                fixed = False
                order = [i, j] if rng.random() < 0.5 else [j, i]
                order.sort(key=lambda p: p in approach)                           # 正在赶往对枪位置的人最后才挪
                for p in order:
                    if not moving[p] or not al[p]:
                        continue
                    enemies = [q for q in (range(5, 10) if p < 5 else range(5)) if al[q] and not ph.pair_allowed(kl, stp, p, q, WIN)]
                    cand, mask = candidates(p)
                    for q in enemies:
                        vis = phys.los_many(map_name, cand[:, 0], cand[:, 1], new[q, 0], new[q, 1])
                        sm = smokes_at(stp)
                        if sm:
                            vis &= ~ph.seg_hits_circles(cand[:, 0], cand[:, 1], new[q, 0], new[q, 1], sm, ph.SMOKE_R)
                        f1 = ph.faces(new[p, 2], cand[:, 0], cand[:, 1], new[q, 0], new[q, 1])
                        f2 = ph.faces(new[q, 2], new[q, 0], new[q, 1], cand[:, 0], cand[:, 1])
                        near = np.hypot(cand[:, 0] - new[q, 0], cand[:, 1] - new[q, 1]) < ph.R_AWARE
                        mask &= ~(vis & (f1 | f2 | near))
                        mp = 0.5 * (cur[p] + cand)                                    # 半秒处
                        mq = 0.5 * (cur[q] + new[q, :2])
                        vis2 = phys.los_many(map_name, mp[:, 0], mp[:, 1], mq[0], mq[1])
                        sm2 = smokes_at(stp - 0.5)
                        if sm2:
                            vis2 &= ~ph.seg_hits_circles(mp[:, 0], mp[:, 1], mq[0], mq[1], sm2, ph.SMOKE_R)
                        g1 = ph.faces(out[k, p, 2], mp[:, 0], mp[:, 1], mq[0], mq[1])
                        g2 = ph.faces(out[k, q, 2], mq[0], mq[1], mp[:, 0], mp[:, 1])
                        near2 = np.hypot(mp[:, 0] - mq[0], mp[:, 1] - mq[1]) < ph.R_AWARE
                        mask &= ~(vis2 & (g1 | g2 | near2))
                    for q in protect.get(p, []):
                        mask &= phys.los_many(map_name, cand[:, 0], cand[:, 1], new[q, 0], new[q, 1])
                    if p in approach:                                             # 时间紧的人不许离目标更远；到早了的人可以先退回掩体等
                        f_, g0, slack = approach[p]
                        mask &= phys.geo_lookup(map_name, f_, cand[:, 0], cand[:, 1]) <= g0 + (10.0 if slack < 1.5 else 230.0 * min(slack - 1.0, 2.0))
                    if mask.any():
                        new[p, :2] = pick(p, cand, mask)
                        st_["contact_fix_move"] += 1
                        fixed = True
                        break
                if fixed:
                    continue
                # 一个人挪不开：两人一起退——从两人各自可走的位置里挑一对互相看不到、也不近的
                if moving[i] and moving[j]:
                    ci, mski = candidates(i)
                    cj, mskj = candidates(j)
                    for p, cc, mm in ((i, ci, mski), (j, cj, mskj)):
                        for q in protect.get(p, []):
                            mm &= phys.los_many(map_name, cc[:, 0], cc[:, 1], new[q, 0], new[q, 1])
                    ti = np.where(mski)[0]
                    tj = np.where(mskj)[0]
                    if len(ti) and len(tj):
                        ti = ti[np.argsort(-P[i][ti])[:50]]
                        tj = tj[np.argsort(-P[j][tj])[:50]]
                        A_ = ci[np.repeat(ti, len(tj))]
                        B_ = cj[np.tile(tj, len(ti))]
                        okp = ~phys.los_many(map_name, A_[:, 0], A_[:, 1], B_[:, 0], B_[:, 1]) | \
                            ((np.hypot(*(A_ - B_).T) >= ph.R_AWARE) & ~ph.faces(new[i, 2], A_[:, 0], A_[:, 1], B_[:, 0], B_[:, 1]) &
                             ~ph.faces(new[j, 2], B_[:, 0], B_[:, 1], A_[:, 0], A_[:, 1]))
                        Am = 0.5 * (cur[i] + A_)
                        Bm = 0.5 * (cur[j] + B_)
                        okp &= ~phys.los_many(map_name, Am[:, 0], Am[:, 1], Bm[:, 0], Bm[:, 1]) | \
                            ((np.hypot(*(Am - Bm).T) >= ph.R_AWARE) & ~ph.faces(out[k, i, 2], Am[:, 0], Am[:, 1], Bm[:, 0], Bm[:, 1]) &
                             ~ph.faces(out[k, j, 2], Bm[:, 0], Bm[:, 1], Am[:, 0], Am[:, 1]))
                        if okp.any():
                            score = np.log(P[i][np.repeat(ti, len(tj))] + 1e-9) + np.log(P[j][np.tile(tj, len(ti))] + 1e-9)
                            b = int(np.argmax(np.where(okp, score, -np.inf)))
                            new[i, :2], new[j, :2] = A_[b], B_[b]
                            st_["contact_fix_joint"] += 1
                            continue
                # 挪不开：把看着对方的人视角转开（选模型认为最可能、且不看向任何不该交火的敌人的朝向）
                for p, q in ((i, j), (j, i)):
                    if not al[p] or not ph.faces(new[p, 2], new[p, 0], new[p, 1], new[q, 0], new[q, 1]):
                        continue
                    enemies = [r for r in (range(5, 10) if p < 5 else range(5)) if al[r] and not ph.pair_allowed(kl, stp, p, r, WIN)]
                    vis_en = [r for r in enemies if phys.los(map_name, new[p, 0], new[p, 1], new[r, 0], new[r, 1])[0]]
                    probs = pyw[p].cpu().numpy()
                    for b in np.argsort(-probs):
                        yw = (b + 0.5) * (360 / NYAW)
                        if not any(ph.faces(yw, new[p, 0], new[p, 1], new[r, 0], new[r, 1]) for r in vis_en):
                            new[p, 2] = yw
                            fixed = True
                            break
                if fixed:
                    st_["contact_fix_yaw"] += 1
                else:
                    st_["contact_fail"] += 1
                    st_.setdefault("contact_fail_detail", []).append(
                        (stp, i, j, float(np.hypot(*(new[i, :2] - new[j, :2]))), i in approach, j in approach, bool(moving[i]), bool(moving[j]),
                         min([abs(t - stp) for t, a_, v_ in kl if a_ in (i, j) or v_ in (i, j)] or [99]),
                         min([t - stp for t, a_, v_ in kl if {a_, v_} == {i, j} and t >= stp] or [99]),
                         stp - max([float(e["time"]) for e in (events or []) if e.get("type") == "end"] or [999]),
                         (i, j) in set(ph.contacts(phys, map_name, new[:, :2], new[:, 2], al))))
                    break
        return out


# ======================================================================== 训练
def transfer_embeddings(model, old_state, old_V, new_V):
    """从旧模型继续训练：形状相同的参数直接复制；选手/战队/点位向量按 steamid、队名、点位名对齐复制（新选手保持初始化）。"""
    own = model.state_dict()
    maps_ = {"pe.weight": (old_V["p"], new_V.p), "pye.weight": (old_V["py"], new_V.py), "tme.weight": (old_V["team"], new_V.team),
             "tmye.weight": (old_V["team_year"], new_V.team_year), "pl_e.weight": (old_V["places"], new_V.places)}
    n_copy = 0
    for k, v in old_state.items():
        if k in maps_:
            ov, nv = maps_[k]
            for key, j in nv.items():
                i = ov.get(key)
                if i is not None and i < v.shape[0] and j < own[k].shape[0]:
                    own[k][j] = v[i]
            n_copy += 1
        elif k in own and own[k].shape == v.shape:
            own[k] = v
            n_copy += 1
    model.load_state_dict(own)
    return n_copy


def train_main(epochs=None, bs=96, lr=3e-4):
    epochs = epochs or int(os.environ.get("CS2_MOTION_EPOCHS", 1 if p4.QUICK else 20))
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    maps, rd, V, A, Bnd_np, tracks = load_data()
    print(f"加载用时 {time.time() - t0:.0f}s；走位数组 {tracks.shape}", flush=True)
    T = torch.as_tensor(np.asarray(tracks), device=DEV)                   # float16，约 1.4 GB
    Bnd = torch.as_tensor(Bnd_np, device=DEV)
    ok = (A["trow"] >= 0) & (A["nsteps"] >= 10)
    tr = np.where(((rd.split.values == "train") | p4.FINAL) & ok)[0]
    va = np.where((rd.split.values == "valid") & ok)[0]
    te = np.where((rd.split.values == "test") & ok)[0]
    model = MotionModel(len(V.p) + 1, len(V.py) + 1, len(V.team) + 1, len(V.team_year) + 1, len(V.places) + 1).to(DEV)
    init = os.environ.get("CS2_MOTION_INIT")                     # 增量更新：从现役走位模型继续训练
    if init and os.path.exists(init):
        old_V = json.load(open(os.path.join(os.path.dirname(init), "vocab.json")))
        n = transfer_embeddings(model, torch.load(init, map_location=DEV), old_V, V)
        lr = lr / 3
        print(f"从 {init} 继续训练，复制 {n} 组参数", flush=True)
    print(f"回合 train {len(tr)} / valid {len(va)} / test {len(te)}；参数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M；设备 {DEV}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)
    rng = np.random.default_rng(0)
    best, bad = 1e9, 0

    def evaluate(idx):
        model.eval()
        r2 = np.random.default_rng(1)
        tot = np.zeros(4)
        n = 0
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                j = idx[i:i + 256]
                out = loss_fn(model, T, Bnd, A, j, r2, train=False)
                tot += np.array([float(out[0]), out[1], out[2], out[3]]) * len(j)
                n += len(j)
        model.train()
        return tot / n

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        te0 = time.time()
        for i in range(0, len(perm), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss, _, _, _ = loss_fn(model, T, Bnd, A, perm[i:i + bs], rng)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate(va)
        print(f"  epoch {ep + 1}: valid 位移 {v[1]:.4f} 朝向 {v[2]:.4f} 位移命中 {v[3]:.3f} | {time.time() - te0:.0f}s", flush=True)
        if v[0] < best - 1e-3 or p4.FINAL:
            best, bad = v[0], 0
            torch.save(model.state_dict(), f"{MDIR}/motion_model.pt")
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(torch.load(f"{MDIR}/motion_model.pt", map_location=DEV))
    vt = evaluate(te)

    # ---- 可走区域、出生点、地图范围、词表
    json.dump(V.to_json(), open(f"{MDIR}/vocab.json", "w"))
    json.dump(Bnd_np.tolist(), open(f"{MDIR}/bounds.json", "w"))
    walk, spawns = {}, {}
    rr = np.random.default_rng(0)
    for mi, m in enumerate(MAPS):
        idx = np.where((A["m"] == mi) & ok & ((rd.split.values == "train") | p4.FINAL))[0]
        if not len(idx):
            continue
        b = Bnd_np[mi]
        nx, ny = int((b[1] - b[0]) / WALK_CELL) + 1, int((b[3] - b[2]) / WALK_CELL) + 1
        H = np.zeros((nx, ny), np.int32)
        for i in np.array_split(rr.choice(idx, size=min(len(idx), 6000), replace=False), 20):
            a = np.asarray(tracks[np.sort(A["trow"][i])], dtype=np.float32)
            al = a[..., 3] > 0
            xi = np.clip(((a[..., 0][al] - b[0]) / WALK_CELL).astype(int), 0, nx - 1)
            yi = np.clip(((a[..., 1][al] - b[2]) / WALK_CELL).astype(int), 0, ny - 1)
            np.add.at(H, (xi, yi), 1)
        walk[m] = H >= 3
        walk[m + "_density"] = H
        sp = []
        for i in rr.choice(idx, size=min(len(idx), 300), replace=False):
            a = np.asarray(tracks[A["trow"][i], 0], dtype=np.float32)
            sd = A["side"][i]
            if (a[:, 3] > 0).all():
                sp.append({"ct": [[float(a[s, 0]), float(a[s, 1]), float(a[s, 2])] for s in range(10) if sd[s] == 1],
                           "t": [[float(a[s, 0]), float(a[s, 1]), float(a[s, 2])] for s in range(10) if sd[s] == 0]})
        spawns[m] = [x for x in sp if len(x["ct"]) == 5 and len(x["t"]) == 5]
    np.savez_compressed(f"{MDIR}/walkable.npz", **walk)
    json.dump(spawns, open(f"{MDIR}/spawns.json", "w"))
    txt = "\n".join([f"# V3b 走位模型报告 {time.strftime('%F %T')} 用时 {time.time() - t0:.0f}s，设备 {DEV}",
                     f"回合 train {len(tr)} / valid {len(va)} / test {len(te)}",
                     f"盲测：位移交叉熵 {vt[1]:.4f}，朝向交叉熵 {vt[2]:.4f}，位移格子命中率 {vt[3]:.3f}（共 {GRID * GRID} 格，均匀猜测命中率 {1 / GRID ** 2:.4f}）"])
    open(f"{MDIR}/report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    train_main() if (len(sys.argv) < 2 or sys.argv[1] == "train") else None
