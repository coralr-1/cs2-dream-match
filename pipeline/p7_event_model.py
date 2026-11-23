"""V2 回合事件生成模型（Transformer）+ V3 整场生成。

用法：
    python p7_event_model.py train       训练、真实度评估、盲测，输出 <模型目录>/report.txt
作为库：
    import p7_event_model as em
    G = em.Generator.load()
    res = G.generate_match([(steamid, year)] * 5, [(steamid, year)] * 5, "de_mirage", n_sims=400)
    res["p_win_A"], res["story"]          # 胜率、一场完整剧本（逐回合事件，可直接用于渲染视频）

路径可用环境变量覆盖（在 Mac 上使用时）：CS2_CLEAN（clean 表目录）、CS2_MODELS（模型目录）。

模型结构：
    上下文 token：1 个回合 token（地图 + 比分/回合进度）+ 10 个选手 token（选手向量 + 年份修正 + 队伍 + 阵营 + 位置号 + 装备）
    事件 token：BOS、击杀、下包、回合结束；每个事件由类型、时间、攻击者/受害者位置号、武器、爆头、点位、包点、结束原因组合而成
    解码：先预测下一事件类型；击杀先用指针网络在存活者中选受害者，再在对方存活者中选攻击者，然后预测武器/爆头/点位/时间
    规则约束只在采样时施加：只能杀存活的敌人、团灭即结束、只有存活 T 能下包且只下一次、下包后只能爆炸/拆除/CT 团灭、时间不倒流
"""
import paths
import json
import math
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import p4_model as p4

C = os.environ.get("CS2_CLEAN", p4.C)
TAG = os.environ.get("CS2_TAG", "")             # 例如 team：结果存到 models/event_team，基线（无战队向量）在 models/event
MDIR = paths.MODELS + "/event" + (f"_{TAG}" if TAG else "")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MAPS = p4.MAPS
TBIN = 2.0                     # 时间桶 2 秒
NT = 80                        # 0–160 秒
MAXEV = 15                     # BOS + 最多 14 个事件
NCTX = 11                      # 基线：回合 token + 10 名选手；战队版再加 2 个战队 token（见 EventModel.nctx）
REASONS = ["t_killed", "ct_killed", "bomb_exploded", "bomb_defused", "time_ran_out"]
REASON_WIN_CT = torch.tensor([1, 0, 0, 1, 1])          # 该原因下 CT 是否获胜
PLANT_MAX_T = 57               # 115 秒后不能再下包
BOMB_T = 19                    # 下包后约 38 秒爆炸
TIME_OUT_T = 52                # 约 104 秒后才可能「时间耗尽」
CFG = {"d": 256, "layers": 6, "heads": 8, "min_rounds": 50, "n_weapons": 30, "team": TAG in ("team", "team2"), "min_team_rounds": 300,
       "team_mod": TAG == "team2"}      # team2：战队向量还会直接叠加到该队每名选手的表示上


# ======================================================================== 数据
def load_tables():
    maps = pd.read_parquet(f"{C}/maps.parquet")
    rd = pd.read_parquet(f"{C}/rounds.parquet", columns=["map_uid", "real_round", "ct_team", "winner_team", "score1_before", "score2_before"])
    rd = rd.merge(maps[["map_uid", "year", "event_id", "map_name", "match_id", "team1_slug", "team2_slug"]], on="map_uid")
    rd = rd[rd.winner_team.isin([1, 2])].reset_index(drop=True)
    rd["split"] = np.where(rd.event_id.isin(p4.TEST_EVENTS), "test", np.where(rd.event_id.isin(p4.VALID_EVENTS), "valid", "train"))
    ex = p4.excluded_maps()
    if ex:
        rd = rd[~rd.map_uid.isin(ex)].reset_index(drop=True)
    sl = pd.read_parquet(f"{C}/round_slots.parquet")
    sl["steamid"] = sl.steamid.astype(str)
    ev = pd.read_parquet(f"{C}/events.parquet")
    ev = ev[~ev.post_end.astype(bool)]
    return maps, rd, sl, ev


class Vocab:
    """选手 / 选手-年份 / 点位 / 武器 的编号，以及每张图允许出现的点位。"""

    def __init__(self, rd, sl, ev):
        tr = rd[rd.split == "train"][["map_uid", "real_round", "year", "map_name"]]
        s = sl.merge(tr, on=["map_uid", "real_round"])
        cp = s.steamid.value_counts()
        cpy = s.groupby(["steamid", "year"]).size()
        self.p = {k: i + 1 for i, k in enumerate(sorted(cp[cp >= CFG["min_rounds"]].index))}
        self.py = {f"{a}|{int(b)}": i + 1 for i, (a, b) in enumerate(sorted(cpy[cpy >= CFG["min_rounds"]].index))}
        e = ev.merge(tr, on=["map_uid", "real_round"])
        k = e[e.type == "kill"]
        w = k.weapon.value_counts()
        self.weapons = {x: i + 2 for i, x in enumerate(w.index[:CFG["n_weapons"]])}         # 0 无，1 其他
        pl = pd.concat([k[["map_name", "a_place"]].set_axis(["map_name", "place"], axis=1), k[["map_name", "v_place"]].set_axis(["map_name", "place"], axis=1)]).dropna()
        pc = pl.groupby(["map_name", "place"]).size()
        pc = pc[pc >= 20]
        self.places = {x: i + 1 for i, x in enumerate(sorted(set(pc.index.get_level_values(1))))}
        self.map_places = {m: sorted(self.places[p] for p in g.index.get_level_values(1)) for m, g in pc.groupby(level=0)}
        trr = rd[rd.split == "train"]
        tms = pd.concat([trr[["team1_slug", "year"]].set_axis(["slug", "year"], axis=1), trr[["team2_slug", "year"]].set_axis(["slug", "year"], axis=1)]).dropna()
        ct, cty = tms.slug.value_counts(), tms.groupby(["slug", "year"]).size()
        self.team = {k: i + 1 for i, k in enumerate(sorted(ct[ct >= CFG["min_team_rounds"]].index))}
        self.team_year = {f"{a}|{int(b)}": i + 1 for i, (a, b) in enumerate(sorted(cty[cty >= CFG["min_team_rounds"]].index))}

    def itm(self, slug):
        return self.team.get(slug, 0) if isinstance(slug, str) else 0

    def itmy(self, slug, y):
        return self.team_year.get(f"{slug}|{int(y)}", 0) if isinstance(slug, str) else 0

    def ip(self, s):
        return self.p.get(s, 0)

    def ipy(self, s, y):
        return self.py.get(f"{s}|{int(y)}", 0)

    def iw(self, w):
        return self.weapons.get(w, 1) if isinstance(w, str) else 0

    def ipl(self, p):
        return self.places.get(p, 0) if isinstance(p, str) else 0

    def to_json(self):
        return {"p": self.p, "py": self.py, "weapons": self.weapons, "places": self.places, "map_places": self.map_places,
                "team": self.team, "team_year": self.team_year}

    @classmethod
    def from_json(cls, d):
        v = cls.__new__(cls)
        v.p, v.py, v.weapons, v.places, v.map_places = d["p"], d["py"], d["weapons"], d["places"], d["map_places"]
        v.team, v.team_year = d.get("team", {}), d.get("team_year", {})
        return v


def build_arrays(rd, sl, ev, V):
    R = len(rd)
    rd = rd.assign(ri=np.arange(R))
    s = sl.merge(rd[["map_uid", "real_round", "ri", "year"]], on=["map_uid", "real_round"])
    A = {"p": np.zeros((R, 10), np.int64), "py": np.zeros((R, 10), np.int64), "side": np.zeros((R, 10), np.int64),
         "eq": np.zeros((R, 10), np.float32), "present": np.zeros((R, 10), np.float32)}
    A["p"][s.ri, s.slot] = [V.ip(x) for x in s.steamid]
    A["py"][s.ri, s.slot] = [V.ipy(x, y) for x, y in zip(s.steamid, s.year)]
    A["side"][s.ri, s.slot] = (s.side.values == "ct").astype(int)
    A["eq"][s.ri, s.slot] = s.equip.values / 1000.0
    A["present"][s.ri, s.slot] = 1.0
    side1 = (rd.ct_team.values == 1).astype(np.float32)
    A["rctx"] = p4.ctx_features(rd.score1_before.values.astype(float), rd.score2_before.values.astype(float), rd.real_round.values, side1)
    A["m"] = np.array([MAPS.index(x) for x in rd.map_name], np.int64)
    A["win1"] = (rd.winner_team.values == 1).astype(np.float32)
    A["tm"] = np.stack([[V.itm(x) for x in rd.team1_slug], [V.itm(x) for x in rd.team2_slug]], 1).astype(np.int64)
    A["tmy"] = np.stack([[V.itmy(x, y) for x, y in zip(rd.team1_slug, rd.year)], [V.itmy(x, y) for x, y in zip(rd.team2_slug, rd.year)]], 1).astype(np.int64)
    e = ev.merge(rd[["map_uid", "real_round", "ri"]], on=["map_uid", "real_round"]).sort_values(["ri", "seq"])
    e["j"] = e.groupby("ri").cumcount() + 1
    e = e[e.j < MAXEV]
    for f in ("type", "t", "a", "v", "w", "hs", "apl", "vpl", "site", "reason"):
        A[f] = np.zeros((R, MAXEV), np.int64)
    A["type"][:, 0] = 1
    A["a"][:], A["v"][:] = 10, 10
    ri, j = e.ri.values, e.j.values
    A["type"][ri, j] = e.type.map({"kill": 2, "plant": 3, "end": 4}).values
    A["t"][ri, j] = np.clip((e.time.values / TBIN).astype(int), 0, NT - 1)
    A["a"][ri, j] = np.where(e.a_slot.values >= 0, e.a_slot.values, 10)
    A["v"][ri, j] = np.where(e.v_slot.values >= 0, e.v_slot.values, 10)
    A["w"][ri, j] = [V.iw(x) for x in e.weapon]
    A["hs"][ri, j] = e.hs.fillna(False).astype(int).values
    A["apl"][ri, j] = [V.ipl(x) for x in e.a_place]
    A["vpl"][ri, j] = [V.ipl(x) for x in e.v_place]
    A["site"][ri, j] = e.site.map({"A": 1, "B": 2}).fillna(0).astype(int).values
    A["reason"][ri, j] = e.reason.map({r: i + 1 for i, r in enumerate(REASONS)}).fillna(0).astype(int).values
    return A


# ======================================================================== 模型
class EventModel(nn.Module):
    def __init__(self, n_p, n_py, n_place, n_weapon, d=CFG["d"], layers=CFG["layers"], heads=CFG["heads"], n_team=0, n_team_year=0, team_mod=False):
        super().__init__()
        self.d = d
        self.use_team = n_team > 0
        self.team_mod = team_mod and self.use_team
        self.nctx = NCTX + (2 if self.use_team else 0)
        if self.use_team:
            self.tme, self.tmye = nn.Embedding(n_team, d), nn.Embedding(n_team_year, d)
            nn.init.normal_(self.tme.weight, std=0.02)
            nn.init.zeros_(self.tmye.weight)
            if team_mod:
                self.team_to_player = nn.Linear(d, d)
        E = nn.Embedding
        self.pe, self.pye = E(n_p, d), E(n_py, d)
        nn.init.normal_(self.pe.weight, std=0.02)
        nn.init.zeros_(self.pye.weight)
        self.team_e, self.side_e, self.slot_e, self.absent_e = E(2, d), E(2, d), E(11, d), E(2, d)
        self.eq_proj = nn.Linear(1, d)
        self.map_e, self.round_proj = E(len(MAPS), d), nn.Linear(7, d)
        self.type_e, self.t_e, self.w_e, self.hs_e = E(5, d), E(NT, d), E(n_weapon, d), E(2, d)
        self.apl_e, self.vpl_e, self.site_e, self.reason_e, self.pos_e = E(n_place, d), E(n_place, d), E(3, d), E(6, d), E(MAXEV, d)
        self.vslot_e = E(11, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        nc = self.nctx
        L = nc + MAXEV
        m = torch.zeros(L, L, dtype=torch.bool)
        m[:nc, nc:] = True                                                  # 上下文看不到事件
        m[nc:, nc:] = torch.triu(torch.ones(MAXEV, MAXEV, dtype=torch.bool), 1)       # 事件只看过去
        self.register_buffer("attn_mask", m, persistent=False)
        self.register_buffer("team_ids", torch.tensor([0] * 5 + [1] * 5), persistent=False)
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))
        self.type_head = nn.Linear(d, 3)
        self.qv, self.kv = nn.Linear(d, d), nn.Linear(d, d)
        self.qa, self.ka = mlp(2 * d, d), nn.Linear(d, d)
        self.qp, self.kp = nn.Linear(d, d), nn.Linear(d, d)
        self.kill_z = mlp(3 * d, d)
        self.w_head, self.hs_head, self.apl_head, self.vpl_head, self.kt_head = nn.Linear(d, n_weapon), nn.Linear(d, 1), nn.Linear(d, n_place), nn.Linear(d, n_place), nn.Linear(d, NT)
        self.plant_z = mlp(2 * d, d)
        self.site_head, self.pt_head = nn.Linear(d, 2), nn.Linear(d, NT)
        self.end_z = mlp(d, d)
        self.reason_head, self.et_head = nn.Linear(d, 5), nn.Linear(d, NT)
        self.win_head = mlp(d, 1)

    def encode(self, b):
        B = b["m"].shape[0]
        slot_ids = torch.arange(10, device=b["m"].device)
        rt = self.map_e(b["m"]) + self.round_proj(b["rctx"])
        pres = b["present"]
        pt = (self.pe(b["p"]) + self.pye(b["py"]) + self.team_e(self.team_ids)[None] + self.side_e(b["side"]) +
              self.slot_e(slot_ids)[None] + self.eq_proj(b["eq"].unsqueeze(-1))) * pres.unsqueeze(-1) + self.absent_e((pres == 0).long())
        ev = (self.type_e(b["type"]) + self.t_e(b["t"]) + self.slot_e(b["a"]) + self.vslot_e(b["v"]) + self.w_e(b["w"]) + self.hs_e(b["hs"]) +
              self.apl_e(b["apl"]) + self.vpl_e(b["vpl"]) + self.site_e(b["site"]) + self.reason_e(b["reason"]) +
              self.pos_e(torch.arange(MAXEV, device=b["m"].device))[None])
        toks = [rt[:, None]]
        if self.use_team:
            two = torch.arange(2, device=x_dev(b))
            tv = self.tme(b["tm"]) + self.tmye(b["tmy"])                      # [B, 2, d] 两队的战队风格
            tt = tv + self.team_e(two)[None] + self.side_e(b["side"][:, [0, 5]])
            toks.append(tt)
            if self.team_mod:                                                # 战队风格直接影响每名选手
                pt = pt + self.team_to_player(tv).repeat_interleave(5, dim=1) * pres.unsqueeze(-1)
        x = torch.cat(toks + [pt, ev], 1)
        pad = torch.cat([torch.zeros(B, self.nctx, dtype=torch.bool, device=x.device), b["type"] == 0], 1)
        return self.norm(self.tf(x, mask=self.attn_mask, src_key_padding_mask=pad))

    @staticmethod
    def gather(S, idx):
        return torch.gather(S, 1, idx.clamp(0, 9).unsqueeze(-1).expand(-1, -1, S.shape[-1]))

    def ptr(self, q, k, S):
        return torch.einsum("bld,bsd->bls", q, k(S)) / math.sqrt(self.d)

    def heads(self, h, S, v_idx, a_idx, p_idx):
        """h: [B,L,d] 事件位置的隐状态；v/a/p_idx: [B,L] 受害者/攻击者/下包者位置号（训练时用真实值）。"""
        Sv, Sa, Sp = self.gather(S, v_idx), self.gather(S, a_idx), self.gather(S, p_idx)
        z = self.kill_z(torch.cat([h, Sa, Sv], -1))
        zp = self.plant_z(torch.cat([h, Sp], -1))
        ze = self.end_z(h)
        return {"type": self.type_head(h), "v": self.ptr(self.qv(h), self.kv, S), "a": self.ptr(self.qa(torch.cat([h, Sv], -1)), self.ka, S),
                "p": self.ptr(self.qp(h), self.kp, S), "w": self.w_head(z), "hs": self.hs_head(z).squeeze(-1), "apl": self.apl_head(z),
                "vpl": self.vpl_head(z), "kt": self.kt_head(z), "site": self.site_head(zp), "pt": self.pt_head(zp),
                "reason": self.reason_head(ze), "et": self.et_head(ze)}


def masked_ce(logits, target, mask, valid):
    """valid 为 False 的位置不计损失；mask 外的类别不参与归一化（真实标签总是保留）。"""
    mask = mask | F.one_hot(target.clamp(0, logits.shape[-1] - 1), logits.shape[-1]).bool()
    lg = logits.masked_fill(~mask, -1e4)
    l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), target.clamp(0, logits.shape[-1] - 1).reshape(-1), reduction="none").view(target.shape)
    return (l * valid).sum(), valid.sum()


def compute_loss(model, b):
    h = model.encode(b)
    S, hev = h[:, model.nctx - 10:model.nctx], h[:, model.nctx:]
    hj = hev[:, :-1]
    tg = {k: b[k][:, 1:] for k in ("type", "t", "a", "v", "w", "hs", "apl", "vpl", "site", "reason")}
    exists = tg["type"] > 0
    is_k, is_p, is_e = tg["type"] == 2, tg["type"] == 3, tg["type"] == 4
    # 每一步之前的存活情况
    vic = F.one_hot(b["v"].clamp(0, 10), 11)[..., :10] * (b["type"] == 2).unsqueeze(-1)
    killed = vic.cumsum(1)[:, :-1]
    alive = (b["present"].unsqueeze(1) > 0) & (killed == 0)
    side = b["side"].unsqueeze(1)
    team = model.team_ids.view(1, 1, 10)
    v_team = (tg["v"] >= 5).long().unsqueeze(-1)
    out = model.heads(hj, S, tg["v"], tg["a"], tg["a"])
    tot, n = {}, {}
    tt = (tg["type"] - 2).clamp(min=0)
    tot["type"], n["type"] = masked_ce(out["type"], tt, torch.ones_like(out["type"], dtype=torch.bool), exists.float())
    kv = is_k & (tg["v"] < 10)
    tot["v"], n["v"] = masked_ce(out["v"], tg["v"], alive, kv.float())
    tot["a"], n["a"] = masked_ce(out["a"], tg["a"], alive & (team != v_team), (kv & (tg["a"] < 10)).float())
    tot["p"], n["p"] = masked_ce(out["p"], tg["a"], alive & (side == 0), (is_p & (tg["a"] < 10)).float())
    full = lambda x: torch.ones_like(x, dtype=torch.bool)
    tot["w"], n["w"] = masked_ce(out["w"], tg["w"], full(out["w"]), is_k.float())
    tot["hs"] = (F.binary_cross_entropy_with_logits(out["hs"], tg["hs"].float(), reduction="none") * is_k).sum()
    n["hs"] = is_k.sum()
    tot["apl"], n["apl"] = masked_ce(out["apl"], tg["apl"], full(out["apl"]), (is_k & (tg["apl"] > 0)).float())
    tot["vpl"], n["vpl"] = masked_ce(out["vpl"], tg["vpl"], full(out["vpl"]), (is_k & (tg["vpl"] > 0)).float())
    tot["kt"], n["kt"] = masked_ce(out["kt"], tg["t"], full(out["kt"]), is_k.float())
    tot["site"], n["site"] = masked_ce(out["site"], (tg["site"] - 1).clamp(min=0), full(out["site"]), (is_p & (tg["site"] > 0)).float())
    tot["pt"], n["pt"] = masked_ce(out["pt"], tg["t"], full(out["pt"]), is_p.float())
    tot["reason"], n["reason"] = masked_ce(out["reason"], (tg["reason"] - 1).clamp(min=0), full(out["reason"]), (is_e & (tg["reason"] > 0)).float())
    tot["et"], n["et"] = masked_ce(out["et"], tg["t"], full(out["et"]), is_e.float())
    win = model.win_head(h[:, 0]).squeeze(-1)
    tot["win"] = F.binary_cross_entropy_with_logits(win, b["win1"], reduction="sum")
    n["win"] = torch.tensor(float(len(win)), device=win.device)
    loss = sum(tot[k] / n[k].clamp(min=1) for k in tot)
    return loss, {k: float(tot[k] / n[k].clamp(min=1)) for k in tot}, win


def x_dev(b):
    return b["m"].device


def to_dev(A, idx):
    return {k: torch.as_tensor(v[idx], device=DEV) for k, v in A.items()}


def train(Atr, Ava, V, epochs=30, bs=512, lr=3e-4, seed=0, patience=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EventModel(len(V.p) + 1, len(V.py) + 1, len(V.places) + 1, CFG["n_weapons"] + 2,
                       n_team=(len(V.team) + 1) if CFG["team"] else 0, n_team_year=(len(V.team_year) + 1) if CFG["team"] else 0,
                       team_mod=CFG["team_mod"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(Atr["m"])
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = to_dev(Atr, perm[i:i + bs])
            drop = torch.rand(b["py"].shape, device=DEV) < 0.2                # 随机去掉年份修正，让通用向量承载主要特征
            b["py"] = torch.where(drop, torch.zeros_like(b["py"]), b["py"])
            if model.use_team:                                               # 遮掉部分战队身份，让模型也会处理「自由组合」的阵容
                dt = torch.rand(b["tm"].shape, device=DEV) < 0.2
                b["tm"] = torch.where(dt, torch.zeros_like(b["tm"]), b["tm"])
                b["tmy"] = torch.where(dt | (torch.rand(b["tmy"].shape, device=DEV) < 0.2), torch.zeros_like(b["tmy"]), b["tmy"])
            loss, _, _ = compute_loss(model, b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vl, parts = evaluate(model, Ava)
        print(f"  epoch {ep + 1}: valid {vl:.4f} | " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()), flush=True)
        if vl < best - 1e-3:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if p4.FINAL:
        return model.eval(), vl
    model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def evaluate(model, A, bs=2048):
    model.eval()
    tot, cnt = Counter(), 0
    for i in range(0, len(A["m"]), bs):
        b = to_dev(A, np.arange(i, min(i + bs, len(A["m"]))))
        loss, parts, _ = compute_loss(model, b)
        w = len(b["m"])
        cnt += w
        for k, v in parts.items():
            tot[k] += v * w
    parts = {k: v / cnt for k, v in tot.items()}
    return sum(parts.values()), parts


@torch.no_grad()
def predict_win(model, A, bs=4096):
    out = []
    for i in range(0, len(A["m"]), bs):
        b = to_dev(A, np.arange(i, min(i + bs, len(A["m"]))))
        out.append(torch.sigmoid(model.win_head(model.encode(b)[:, 0]).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


# ======================================================================== 采样：生成一回合
def sample_masked(logits, mask, gen=None):
    lg = logits.masked_fill(~mask, -1e9)
    return torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(-1)


@torch.no_grad()
def sample_rounds(model, ctx, place_mask, gen=None):
    """ctx: 当前回合上下文张量（每个模拟一行）。返回事件字段 [N, MAXEV]、获胜方是否为 CT。"""
    N = ctx["m"].shape[0]
    dev = ctx["m"].device
    ev = {f: torch.zeros(N, MAXEV, dtype=torch.long, device=dev) for f in ("type", "t", "a", "v", "w", "hs", "apl", "vpl", "site", "reason")}
    ev["type"][:, 0] = 1
    ev["a"][:], ev["v"][:] = 10, 10
    alive = ctx["present"] > 0
    side = ctx["side"]
    team = model.team_ids.view(1, 10)
    planted = torch.zeros(N, dtype=torch.bool, device=dev)
    plant_t = torch.zeros(N, dtype=torch.long, device=dev)
    cur = torch.zeros(N, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    win_ct = torch.zeros(N, dtype=torch.bool, device=dev)
    tgrid = torch.arange(NT, device=dev).view(1, NT)
    pm = place_mask[ctx["m"]]
    ar = torch.arange(N, device=dev)
    for j in range(MAXEV - 1):
        h = model.encode({**ctx, **ev})
        S, hj = h[:, model.nctx - 10:model.nctx], h[:, model.nctx + j]
        ct_alive = (alive & (side == 1)).any(1)
        t_alive = (alive & (side == 0)).any(1)
        valid_reason = torch.stack([~t_alive & ~planted, ~ct_alive, planted, planted & ct_alive, ~planted], 1)
        force_end = (~t_alive & ~planted) | ~ct_alive | (j == MAXEV - 2)
        can_kill = ct_alive & t_alive & ~force_end
        can_plant = ~planted & t_alive & (cur <= PLANT_MAX_T) & ~force_end
        can_end = valid_reason.any(1) & (force_end | planted | (cur >= TIME_OUT_T))
        tmask = torch.stack([can_kill, can_plant, can_end | ~(can_kill | can_plant)], 1)
        H = lambda x: x.unsqueeze(1)
        typ = sample_masked(model.type_head(hj), tmask, gen)
        # 击杀：受害者 → 攻击者 → 细节
        v_logits = model.ptr(H(model.qv(hj)), model.kv, S).squeeze(1)
        v = sample_masked(v_logits, alive | ~(alive.any(1, keepdim=True)), gen)
        Sv = S[ar, v]
        a_logits = model.ptr(H(model.qa(torch.cat([hj, Sv], -1))), model.ka, S).squeeze(1)
        amask = alive & (team != (v >= 5).long().unsqueeze(1))
        a = sample_masked(a_logits, amask | ~amask.any(1, keepdim=True), gen)
        z = model.kill_z(torch.cat([hj, S[ar, a], Sv], -1))
        w = sample_masked(model.w_head(z), torch.ones_like(model.w_head(z), dtype=torch.bool), gen)
        hs = torch.bernoulli(torch.sigmoid(model.hs_head(z).squeeze(-1)), generator=gen).long()
        apl = sample_masked(model.apl_head(z), pm, gen)
        vpl = sample_masked(model.vpl_head(z), pm, gen)
        ktm = (tgrid >= cur.unsqueeze(1)) & ((tgrid <= (plant_t + BOMB_T).unsqueeze(1)) | ~planted.unsqueeze(1)) & ((tgrid <= PLANT_MAX_T + 1) | planted.unsqueeze(1))
        kt = sample_masked(model.kt_head(z), ktm | ~ktm.any(1, keepdim=True), gen)
        # 下包
        pmask = alive & (side == 0)
        p = sample_masked(model.ptr(H(model.qp(hj)), model.kp, S).squeeze(1), pmask | ~pmask.any(1, keepdim=True), gen)
        zp = model.plant_z(torch.cat([hj, S[ar, p]], -1))
        site = sample_masked(model.site_head(zp), torch.ones(N, 2, dtype=torch.bool, device=dev), gen)
        ptm = (tgrid >= cur.unsqueeze(1)) & (tgrid <= PLANT_MAX_T)
        pt = sample_masked(model.pt_head(zp), ptm | ~ptm.any(1, keepdim=True), gen)
        # 回合结束
        ze = model.end_z(hj)
        reason = sample_masked(model.reason_head(ze), valid_reason | ~valid_reason.any(1, keepdim=True), gen)
        elim = (reason <= 1)
        etm = (tgrid >= cur.unsqueeze(1))
        etm = etm & torch.where((reason == 4).unsqueeze(1), tgrid >= TIME_OUT_T, torch.ones_like(etm))
        etm = etm & torch.where((reason == 2).unsqueeze(1), tgrid >= (plant_t + BOMB_T).unsqueeze(1), torch.ones_like(etm))
        et = sample_masked(model.et_head(ze), etm | ~etm.any(1, keepdim=True), gen)
        et = torch.where(elim, cur, et)
        # 写入第 j+1 个事件
        act = ~done
        is_k, is_p, is_e = act & (typ == 0), act & (typ == 1), act & (typ == 2)
        J = j + 1
        ev["type"][:, J] = torch.where(is_k, 2, torch.where(is_p, 3, torch.where(is_e, 4, 0)))
        ev["t"][:, J] = torch.where(is_k, kt, torch.where(is_p, pt, torch.where(is_e, et, 0)))
        ev["v"][:, J] = torch.where(is_k, v, 10)
        ev["a"][:, J] = torch.where(is_k, a, torch.where(is_p, p, 10))
        ev["w"][:, J] = torch.where(is_k, w, 0)
        ev["hs"][:, J] = torch.where(is_k, hs, 0)
        ev["apl"][:, J] = torch.where(is_k, apl, 0)
        ev["vpl"][:, J] = torch.where(is_k, vpl, 0)
        ev["site"][:, J] = torch.where(is_p, site + 1, 0)
        ev["reason"][:, J] = torch.where(is_e, reason + 1, 0)
        alive[ar[is_k], v[is_k]] = False
        planted = planted | is_p
        plant_t = torch.where(is_p, pt, plant_t)
        cur = torch.maximum(cur, ev["t"][:, J] * act)
        win_ct = torch.where(is_e, REASON_WIN_CT.to(dev)[reason].bool(), win_ct)
        done = done | is_e
        if bool(done.all()):
            break
    return {k: v.cpu().numpy() for k, v in ev.items()}, win_ct.cpu().numpy()


# ======================================================================== 整场生成
class Generator:
    def __init__(self, model, V, pool, eq_rank, place_xy, names):
        self.model, self.V, self.pool, self.eq_rank, self.place_xy, self.names = model.eval(), V, pool, eq_rank, place_xy, names
        self.phys = None          # p13 地图物理（可选）：有它时击杀双方的位置从真实对枪里成对抽取，保证互相看得见、距离合理
        n_place = len(V.places) + 1
        pm = torch.zeros(len(MAPS), n_place, dtype=torch.bool)
        for m, ids in V.map_places.items():
            pm[MAPS.index(m), ids] = True
        pm[:, 0] = ~pm[:, 1:].any(1)
        self.place_mask = pm.to(DEV)
        self.id2place = {i: p for p, i in V.places.items()}
        self.id2weapon = {i: w for w, i in V.weapons.items()}
        self.id2weapon[1] = "other"

    @classmethod
    def load(cls, mdir=MDIR):
        V = Vocab.from_json(json.load(open(f"{mdir}/vocab.json")))
        cfg = json.load(open(f"{mdir}/config.json"))
        nt = (len(V.team) + 1) if cfg.get("team") else 0
        model = EventModel(len(V.p) + 1, len(V.py) + 1, len(V.places) + 1, cfg["n_weapons"] + 2, d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"],
                           n_team=nt, n_team_year=(len(V.team_year) + 1) if nt else 0, team_mod=cfg.get("team_mod", False)).to(DEV)
        model.load_state_dict(torch.load(f"{mdir}/event_model.pt", map_location=DEV))
        pool = p4.EconPool.from_dict(torch.load(f"{mdir}/econ_pool.pt", weights_only=False))
        return cls(model, V, pool, json.load(open(f"{mdir}/equip_rank.json")), json.load(open(f"{mdir}/place_xy.json")),
                   json.load(open(f"{mdir}/names.json")))

    def generate_match(self, teamA, teamB, map_name, n_sims=400, seed=0, a_starts_ct=None, story="typical", team_ids=(None, None),
                       guide=None, cands=4, progress=None, winner=None):
        """teamA/teamB: [(steamid, year)] × 5；team_ids: ((slug, year) 或 None, ...)，None 表示自由组合阵容。
        guide: P4 模拟器（p4.Simulator）。提供时每回合先用 P4 回合模型抽出胜方，再从事件模型生成的 cands 个候选里
               挑胜负一致的那个——胜负概率与 P4 一样准，过程仍由事件模型生成。
        返回胜率、比分分布、按比分的实时胜率表和一场完整剧本。"""
        gen = torch.Generator(device=DEV).manual_seed(seed)
        rng = np.random.default_rng(seed)
        N = n_sims
        teamA = sorted(teamA, key=lambda x: -self.eq_rank.get(x[0], 0))
        teamB = sorted(teamB, key=lambda x: -self.eq_rank.get(x[0], 0))
        roster = teamA + teamB
        P = torch.tensor([self.V.ip(s) for s, _ in roster], device=DEV).repeat(N, 1)
        PY = torch.tensor([self.V.ipy(s, y) for s, y in roster], device=DEV).repeat(N, 1)
        TM = torch.tensor([self.V.itm(t[0]) if t else 0 for t in team_ids], device=DEV).repeat(N, 1)
        TMY = torch.tensor([self.V.itmy(t[0], t[1]) if t else 0 for t in team_ids], device=DEV).repeat(N, 1)
        if guide is not None:
            gdev = p4.DEV
            GP = torch.tensor([guide.vocab.ip(s) for s, _ in roster], device=gdev).repeat(N, 1)
            GPY = torch.tensor([guide.vocab.ipy(s, y) for s, y in roster], device=gdev).repeat(N, 1)
            gm = torch.full((N,), MAPS.index(map_name), device=gdev)
        a_ct_first = rng.random(N) < 0.5 if a_starts_ct is None else np.full(N, bool(a_starts_ct))
        sa, sb = np.zeros(N, int), np.zeros(N, int)
        st = {t: {"win": np.array([None] * N, object), "streak": np.zeros(N, int), "eq": np.full(N, np.nan)} for t in "AB"}
        active = np.ones(N, bool)
        log = []                                          # 每回合：所有模拟的事件、装备、阵营
        rr = 0
        while active.any() and rr < 60:
            rr += 1
            if progress is not None:
                progress(rr)
            if rr <= 12:
                a_ct = a_ct_first.copy()
            elif rr <= 24:
                a_ct = ~a_ct_first
            else:
                a_ct = ~a_ct_first if ((rr - 22) // 6) % 2 == 0 else a_ct_first.copy()
            eqs = np.zeros((N, 10), np.float32)
            for ti, t in enumerate("AB"):
                keys = [p4.econ_key(rr, st[t]["win"][i], st[t]["streak"][i], st[t]["eq"][i]) for i in range(N)]
                groups = {}
                for i, k in enumerate(keys):
                    groups.setdefault(k, []).append(i)
                for k, idx in groups.items():
                    eqs[idx, ti * 5:(ti + 1) * 5] = self.pool.sample(k, len(idx), rng)
            side = np.zeros((N, 10), np.int64)
            side[:, :5] = a_ct[:, None]
            side[:, 5:] = ~a_ct[:, None]
            # 只对还没打完的模拟跑模型（加时长尾里往往只剩几场，全量计算会白白多花一半时间）
            ia = np.where(active)[0]
            na = len(ia)
            ia_t = torch.as_tensor(ia, device=DEV)
            sa_, sb_, act_ = sa[ia].astype(float), sb[ia].astype(float), a_ct[ia]
            ctx = {"m": torch.full((na,), MAPS.index(map_name), device=DEV), "p": P[ia_t], "py": PY[ia_t], "tm": TM[ia_t], "tmy": TMY[ia_t],
                   "side": torch.as_tensor(side[ia], device=DEV), "eq": torch.as_tensor(eqs[ia] / 1000.0, device=DEV),
                   "present": torch.ones(na, 10, device=DEV),
                   "rctx": torch.as_tensor(p4.ctx_features(sa_, sb_, np.full(na, rr), act_.astype(np.float32)), device=DEV)}
            if guide is None:
                ev_a, win_a = sample_rounds(self.model, ctx, self.place_mask, gen)
            else:
                gi = torch.as_tensor(ia, device=gdev)
                with torch.no_grad():
                    e_ = torch.as_tensor(eqs[ia] / 1000.0, device=gdev)
                    s1 = torch.as_tensor(act_.astype(np.float32), device=gdev)
                    c1 = torch.as_tensor(p4.ctx_features(sa_, sb_, np.full(na, rr), act_.astype(np.float32)), device=gdev)
                    c2 = torch.as_tensor(p4.ctx_features(sb_, sa_, np.full(na, rr), (~act_).astype(np.float32)), device=gdev)
                    gp, gpy = GP[gi], GPY[gi]
                    pa = torch.sigmoid(guide.model(gp[:, :5], gpy[:, :5], e_[:, :5], gp[:, 5:], gpy[:, 5:], e_[:, 5:], s1, c1, c2, gm[:na])).cpu().numpy()
                target = rng.random(na) < pa
                big = {k: v.repeat_interleave(cands, 0) for k, v in ctx.items()}
                evb, wctb = sample_rounds(self.model, big, self.place_mask, gen)
                a_won_b = (wctb == np.repeat(act_, cands)).reshape(na, cands)
                match = a_won_b == target[:, None]
                pick_c = np.where(match.any(1), match.argmax(1), 0)
                sel = np.arange(na) * cands + pick_c
                ev_a = {k: v[sel] for k, v in evb.items()}
                win_a = wctb[sel]
            ev = {k: np.zeros((N,) + v.shape[1:], v.dtype) for k, v in ev_a.items()}      # 已结束的模拟填 0（后面只读 active 的行）
            for k, v in ev_a.items():
                ev[k][ia] = v
            win_ct = np.zeros(N, bool)
            win_ct[ia] = win_a
            a_won = (win_ct == a_ct) & active
            b_won = (win_ct != a_ct) & active
            log.append({"ev": ev, "eq": eqs, "a_ct": a_ct.copy(), "active": active.copy(), "score_before": (sa.copy(), sb.copy())})
            sa += a_won
            sb += b_won
            for t, won in (("A", a_won), ("B", b_won)):
                upd = active
                st[t]["eq"][upd] = eqs[upd][:, (0 if t == "A" else 5):(5 if t == "A" else 10)].mean(1)
                st[t]["win"][upd] = won[upd]
                st[t]["streak"][upd] = np.where(won[upd], 0, np.minimum(st[t]["streak"][upd] + 1, 4))
            active = active & ~np.array([p4.match_over(a, b) for a, b in zip(sa, sb)])
        winA = sa > sb
        scores = Counter(f"{a}-{b}" for a, b in zip(sa, sb))
        # 实时胜率：所有模拟中经过某个比分的，最终 A 赢的比例
        through = {}
        for r in log:
            s0, s1 = r["score_before"]
            for i in np.where(r["active"])[0]:
                through.setdefault((int(s0[i]), int(s1[i])), []).append(winA[i])
        live = {f"{a}-{b}": float(np.mean(v)) for (a, b), v in through.items() if len(v) >= 5}
        # 选剧本：默认取「多数赢家 + 最常见比分」的那一场；dramatic 取打到加时、最接近的一场
        maj = winA.mean() >= 0.5 if winner is None else winner == "A"      # winner：系列赛里先抽好这张图谁赢，再挑对应剧本
        cand = np.where(winA == maj)[0]
        if len(cand) == 0:
            cand = np.arange(N)
        if isinstance(story, tuple) and story[0] == "player":                # 选手高光：挑这名选手发挥最好的一场（多杀多、他的队赢）
            si = roster.index(tuple(story[1]))
            kills, multi = np.zeros(N), np.zeros(N)
            for r in log:
                k = ((r["ev"]["type"] == 2) & (r["ev"]["a"] == si)).sum(1) * r["active"]
                kills += k
                multi += k >= 3
            team_won = winA if si < 5 else ~winA
            score = kills + 4 * multi + 6 * team_won
            pick = int(np.argmax(score + rng.uniform(0, 0.5, N)))
        elif story == "dramatic":
            gap = np.abs(sa - sb) + (sa + sb <= 24) * 100
            pick = int(cand[np.argmin(gap[cand])])
        else:
            top = Counter(f"{sa[i]}-{sb[i]}" for i in cand).most_common(1)[0][0]
            pick = int(next(i for i in cand if f"{sa[i]}-{sb[i]}" == top))
        return {"p_win_A": float(winA.mean()), "scores": {k: v / N for k, v in scores.most_common(10)}, "live_win_A": live,
                "story": self._story(log, pick, roster, map_name, rng)}

    def _story(self, log, i, roster, map_name, rng):
        names = [self.names.get(s, s) for s, _ in roster]
        dup = {n for n in names if names.count(n) > 1}          # 同一人以不同年份分在两边时，名字后加年份区分
        names = [f"{n}'{str(y)[-2:]}" if n in dup else n for n, (_, y) in zip(names, roster)]
        rounds, stats = [], [Counter() for _ in range(10)]
        for rr, r in enumerate(log, start=1):
            if not r["active"][i]:
                break
            ev = {k: v[i] for k, v in r["ev"].items()}
            a_ct = bool(r["a_ct"][i])
            events, alive = [], set(range(10))
            first = True
            last_xy = {}                                                     # 每人上一次交火的计划位置 (时间, x, y)：连续交火不能瞬移
            for j in range(1, MAXEV):
                t = int(ev["type"][j])
                if t == 0:
                    break
                sec = float(ev["t"][j] * TBIN + rng.uniform(0, TBIN))
                if t == 2:
                    a, v = int(ev["a"][j]), int(ev["v"][j])
                    ap, vp = self.id2place.get(int(ev["apl"][j])), self.id2place.get(int(ev["vpl"][j]))
                    wpn = self.id2weapon.get(int(ev["w"][j]), "other")
                    pair = None
                    if self.phys is not None:
                        import p13_physics as ph
                        if ph.needs_los(wpn):
                            near = {s_: (lx, ly, 230.0 * (sec - lt) + 150.0) for s_, (lt, lx, ly) in last_xy.items() if sec - lt < 20}
                            pair = self.phys.sample_pair(map_name, ap, vp, wpn, rng, near_a=near.get(a), near_v=near.get(v))
                    axy, vxy = pair if pair else (self._xy(map_name, ap, rng), self._xy(map_name, vp, rng))
                    if axy:
                        last_xy[a] = (sec, axy[0], axy[1])
                    if vxy:
                        last_xy[v] = (sec, vxy[0], vxy[1])
                    e = {"type": "kill", "time": round(sec, 1), "attacker": a, "victim": v, "attacker_name": names[a], "victim_name": names[v],
                         "weapon": wpn, "headshot": bool(ev["hs"][j]), "attacker_place": ap, "victim_place": vp,
                         "attacker_xy": axy, "victim_xy": vxy, "opening": first}
                    first = False
                    alive.discard(v)
                    stats[a]["kills"] += 1
                    stats[a]["hs"] += int(ev["hs"][j])
                    stats[a]["awp_kills"] += int(e["weapon"] == "awp")
                    stats[a]["opening_kills"] += int(e["opening"])
                    stats[v]["deaths"] += 1
                    events.append(e)
                elif t == 3:
                    p = int(ev["a"][j])
                    site = "A" if ev["site"][j] == 1 else "B"
                    events.append({"type": "plant", "time": round(sec, 1), "player": p, "player_name": names[p], "site": site,
                                   "xy": self._xy(map_name, f"Bombsite{site}", rng)})
                elif t == 4:
                    reason = REASONS[int(ev["reason"][j]) - 1]
                    win_ct = bool(REASON_WIN_CT[int(ev["reason"][j]) - 1])
                    winner = "A" if win_ct == a_ct else "B"
                    events.append({"type": "end", "time": round(sec, 1), "reason": reason, "winner": winner})
            # 残局：某队只剩 1 人且该队最终获胜
            winner = events[-1]["winner"] if events and events[-1]["type"] == "end" else None
            for tm, rng_slots in (("A", range(5)), ("B", range(5, 10))):
                left = [s for s in rng_slots if s in alive]
                if winner == tm and len(left) == 1:
                    # 判断是否曾经以 1 打多
                    seq_alive = set(rng_slots)
                    other = set(range(10)) - set(rng_slots)
                    vs = 0
                    for e in events:
                        if e["type"] == "kill":
                            (seq_alive if e["victim"] in seq_alive else other).discard(e["victim"])
                            if len(seq_alive) == 1 and vs == 0:
                                vs = len(other)
                    if vs >= 1:
                        stats[left[0]]["clutch_wins"] += 1
                        events[-1]["clutch"] = {"player": left[0], "player_name": names[left[0]], "vs": vs}
            for s in range(10):
                stats[s]["rounds"] += 1
            s0, s1 = r["score_before"]
            rounds.append({"round": rr, "a_side": "CT" if a_ct else "T", "score_before": [int(s0[i]), int(s1[i])],
                           "equip": [int(x) for x in r["eq"][i]], "events": events, "winner": winner})
        players = []
        for s in range(10):
            st = stats[s]
            n = max(st["rounds"], 1)
            players.append({"slot": s, "team": "A" if s < 5 else "B", "steamid": roster[s][0], "year": roster[s][1], "name": names[s],
                            "kills": st["kills"], "deaths": st["deaths"], "hs_pct": st["hs"] / max(st["kills"], 1), "awp_kills": st["awp_kills"],
                            "opening_kills": st["opening_kills"], "clutch_wins": st["clutch_wins"], "kpr": st["kills"] / n})
        fa = sum(1 for r in rounds if r["winner"] == "A")
        fb = sum(1 for r in rounds if r["winner"] == "B")
        win_team = "A" if fa > fb else "B"
        mvp = max((p for p in players if p["team"] == win_team), key=lambda p: p["kills"] - p["deaths"] * 0.6 + p["opening_kills"] * 0.5 + p["clutch_wins"] * 1.5)
        return {"map": map_name, "final": [fa, fb], "rounds": rounds, "players": players, "mvp": mvp["slot"]}

    def _xy(self, map_name, place, rng):
        pts = self.place_xy.get(map_name, {}).get(place) if place else None
        if not pts:
            return None
        x, y = pts[rng.integers(len(pts))]
        return [x, y]


# ======================================================================== 训练 + 评估
def train_main():
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    maps, rd, sl, ev = load_tables()
    V = Vocab(rd, sl, ev)
    A = build_arrays(rd, sl, ev, V)
    idx = {s: np.where(rd.split.values == s)[0] for s in ("train", "valid", "test")}
    if p4.FINAL:
        idx["train"] = np.arange(len(rd))
    Atr, Ava, Ate = ({k: v[i] for k, v in A.items()} for i in (idx["train"], idx["valid"], idx["test"]))
    print(f"回合 train {len(idx['train'])} / valid {len(idx['valid'])} / test {len(idx['test'])}；选手 {len(V.p)}，选手-年份 {len(V.py)}，点位 {len(V.places)}，设备 {DEV}", flush=True)
    model, best = train(Atr, Ava, V, epochs=1 if p4.QUICK else (11 if p4.FINAL else 30), patience=99 if p4.FINAL else 4)
    torch.save(model.state_dict(), f"{MDIR}/event_model.pt")
    json.dump(V.to_json(), open(f"{MDIR}/vocab.json", "w"))
    json.dump(CFG, open(f"{MDIR}/config.json", "w"))
    # 经济池、装备排序、点位坐标、名字（渲染用）
    S4 = p4.Simulator.load()
    torch.save(S4.pool.to_dict(), f"{MDIR}/econ_pool.pt")
    json.dump(S4.eq_rank, open(f"{MDIR}/equip_rank.json", "w"))
    k = ev[ev.type == "kill"].merge(maps[["map_uid", "map_name"]], on="map_uid")
    pts = pd.concat([k[["map_name", "v_place", "vx", "vy"]].set_axis(["map_name", "place", "x", "y"], axis=1),
                     k[["map_name", "a_place", "ax", "ay"]].set_axis(["map_name", "place", "x", "y"], axis=1)]).dropna()
    place_xy = {}
    for (m, p), g in pts.groupby(["map_name", "place"]):
        g = g.sample(min(len(g), 200), random_state=0)
        place_xy.setdefault(m, {})[p] = [[round(float(x), 1), round(float(y), 1)] for x, y in zip(g.x, g.y)]
    json.dump(place_xy, open(f"{MDIR}/place_xy.json", "w"))
    pl = pd.read_parquet(f"{C}/players.parquet")
    json.dump(dict(zip(pl.steamid.astype(str), pl.name)), open(f"{MDIR}/names.json", "w"), ensure_ascii=False)
    if p4.FINAL:
        print("正式事件模型已保存到", MDIR, flush=True)
        return

    # ---- 评估 1：回合胜负（直接头）
    p_win = predict_win(model, Ate)
    ll = p4.logloss(p_win, Ate["win1"])
    # ---- 评估 2：真实度——以盲测回合的真实阵容/装备/比分为条件各生成一次，对比统计
    G = Generator.load()
    gen = torch.Generator(device=DEV).manual_seed(0)
    sim = []
    for i in range(0, len(Ate["m"]), 1024):
        b = to_dev({k: Ate[k] for k in ("m", "p", "py", "tm", "tmy", "side", "eq", "present", "rctx")}, np.arange(i, min(i + 1024, len(Ate["m"]))))
        e, wct = sample_rounds(model, b, G.place_mask, gen)
        sim.append((e, wct, b["side"][:, 0].cpu().numpy()))
    E = {k: np.concatenate([s[0][k] for s in sim]) for k in sim[0][0]}
    win_ct = np.concatenate([s[1] for s in sim])
    s1ct = np.concatenate([s[2] for s in sim]) == 1
    real = {k: Ate[k] for k in ("type", "t", "a", "v", "w", "hs", "site", "reason")}

    def stats(D):
        kill = D["type"] == 2
        k_round = kill.sum(1)
        ends = D["type"] == 4
        reason = D["reason"][ends]
        aw = V.weapons.get("awp", -1)
        first_k = np.argmax(kill, 1)
        has_k = kill.any(1)
        return {"每回合击杀": round(float(k_round.mean()), 2), "下包率": round(float((D["type"] == 3).any(1).mean()), 3),
                "A 点占比": round(float((D["site"] == 1).sum() / max((D["site"] > 0).sum(), 1)), 3),
                "爆头率": round(float(D["hs"][kill].mean()), 3), "AWP 击杀占比": round(float((D["w"][kill] == aw).mean()), 3),
                "首杀时间中位数(秒)": float(np.median(D["t"][np.arange(len(first_k))[has_k], first_k[has_k]]) * TBIN),
                "回合时长中位数(秒)": float(np.median(D["t"][ends]) * TBIN),
                "结束原因": {REASONS[r - 1]: round(float((reason == r).mean()), 3) for r in range(1, 6)}}
    st_real, st_sim = stats(real), stats(E)
    # 每名选手（按位置号对应到真实选手）的击杀率 / 首杀率 / AWP 占比：生成 vs 真实
    sids = np.array(list(V.p.keys()))
    def per_player(D):
        kill = D["type"] == 2
        rows = []
        first_j = np.argmax(kill, 1)
        for i in range(len(D["type"])):
            ks = kill[i]
            for jj in np.where(ks)[0]:
                rows.append((i, int(D["a"][i, jj]), int(D["w"][i, jj] == V.weapons.get("awp", -1)), int(jj == first_j[i])))
        return pd.DataFrame(rows, columns=["ri", "slot", "awp", "opening"])
    pr_real, pr_sim = per_player(real), per_player(E)
    pid = Ate["p"]
    def agg(df):
        df = df[df.slot < 10]
        df = df.assign(pid=pid[df.ri.values, df.slot.values])
        return df.groupby("pid").agg(kills=("ri", "size"), awp=("awp", "mean"), opening=("opening", "sum"))
    appear = pd.Series(pid[Ate["present"] > 0]).value_counts()
    cr, cs = agg(pr_real), agg(pr_sim)
    j = cr.join(cs, lsuffix="_real", rsuffix="_sim", how="inner").join(appear.rename("rounds"))
    j = j[(j.index > 0) & (j.rounds >= 150)]
    corr = {"击杀/回合": round(float(np.corrcoef(j.kills_real / j.rounds, j.kills_sim / j.rounds)[0, 1]), 3),
            "AWP 占比": round(float(np.corrcoef(j.awp_real, j.awp_sim)[0, 1]), 3),
            "首杀/回合": round(float(np.corrcoef(j.opening_real / j.rounds, j.opening_sim / j.rounds)[0, 1]), 3)}
    gen_win1 = (win_ct == s1ct).astype(float)
    # ---- 评估 3：整场盲测
    res = []
    for r in maps[maps.event_id.isin(p4.TEST_EVENTS) & maps.legal.astype(bool)].itertuples(index=False):
        A_ = [(s, int(r.year)) for s in str(r.roster1).split(",") if s]
        B_ = [(s, int(r.year)) for s in str(r.roster2).split(",") if s]
        if len(A_) != 5 or len(B_) != 5:
            continue
        out = G.generate_match(A_, B_, r.map_name, n_sims=200, seed=zlib.crc32(r.map_uid.encode()),
                               team_ids=((r.team1_slug, int(r.year)), (r.team2_slug, int(r.year))))
        res.append({"map_uid": r.map_uid, "p": out["p_win_A"], "y": float(r.score1 > r.score2)})
    res = pd.DataFrame(res)
    old = pd.read_csv(f"{p4.MDIR}/test_predictions.csv")[["map_uid", "p_model", "p_elo"]]
    res = res.merge(old, on="map_uid", how="left")
    mm = lambda p, y: {"logloss": round(p4.logloss(p, y), 4), "brier": round(float(np.mean((p - y) ** 2)), 4), "acc": round(float(np.mean((p > .5) == (y > .5))), 4)}

    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P(f"# V2 事件生成模型报告（{('战队版 v2：战队向量叠加到选手' if CFG['team_mod'] else '战队版') if CFG['team'] else '基线：无战队向量'}）", time.strftime("%F %T"), f"用时 {time.time() - t0:.0f}s，设备 {DEV}")
    P(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M；valid 总损失 {best:.4f}")
    P(f"\n## 回合胜负（盲测 {len(Ate['m'])} 回合，对数损失）")
    P(f"  事件模型直接头 {ll:.4f}；P4 模拟器的回合模型 0.5970；只看装备 0.6069")
    P(f"  逐事件生成得到的胜负：生成胜率 {gen_win1.mean():.3f} vs 真实 {Ate['win1'].mean():.3f}；与真实一致 {np.mean(gen_win1 == Ate['win1']):.3f}")
    P("\n## 真实度（以盲测回合的真实阵容和装备为条件各生成一次）")
    for k in st_real:
        P(f"  {k}: 生成 {st_sim[k]} | 真实 {st_real[k]}")
    P(f"  每名选手的生成 vs 真实相关系数（{len(j)} 名盲测出场 ≥150 回合的选手）: {corr}")
    P(f"\n## 整场盲测（{len(res)} 张图）")
    P(f"  事件模型整场生成 {mm(res.p.values, res.y.values)}")
    P(f"  P4 模拟器（校准后） {mm(res.p_model.values, res.y.values)}")
    P(f"  战队 Elo {mm(res.p_elo.values, res.y.values)}")
    txt = "\n".join(L)
    open(f"{MDIR}/report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    train_main() if (len(sys.argv) < 2 or sys.argv[1] == "train") else None
