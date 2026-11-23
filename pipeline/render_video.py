"""V4 2D 对战短视频：模拟一场比赛并渲染成 9:16 竖屏 MP4。

单条：
    python render_video.py --a vitality:2025 --b spirit:2026 --map de_mirage --out out/vit25_vs_spi26.mp4
自由组合阵容（每人 名字:年份，名字取最常用名）：
    python render_video.py --a-players "ZywOo:2025,ropz:2025,..." --b-players "donk:2026,..." --map de_inferno --out out/x.mp4
批量（CSV 列：a,b,map[,seed,story]，a/b 写 战队:年份；map 写 bo1/bo3/bo5 时生成含 Ban/Pick 的系列赛）：
    python render_video.py --batch matchups.csv --outdir out/
系列赛（先模拟 Ban/Pick，再按选出的地图逐图打完）：
    python render_video.py --a vitality:2025 --b spirit:2026 --series bo3 --out out/bo3.mp4 [--pool 0] [--first A]

每条视频同时输出：同名 .png 封面、同名 .json 剧本（逐回合事件、比分、数据，方便写文案）。
画面全程标注「AI 模拟对局 · 非真实比赛」。
"""
import paths
import argparse
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch

import p4_model as p4
import p7_event_model as em
import p10_motion as mo
import p12_veto as vt
import p13_physics as ph
from matplotlib.patches import Circle

C = em.C
FPS = 30
SPEED = 8.0                       # 关键回合的播放倍速
SUMMARY_FRAMES = 24               # 普通回合每回合帧数（0.8 秒）
INTRO_FRAMES, OUTRO_FRAMES = 84, 132
BG, PANEL, TXT, DIM = "#0e1116", "#161b23", "#eceff4", "#8b93a1"
COL = {"A": "#f5a623", "B": "#3fa9ff"}
WEAPON_CN = {"awp": "AWP", "ak47": "AK-47", "m4a1_silencer": "M4A1-S", "m4a1": "M4A4", "deagle": "沙鹰", "galilar": "加利尔", "famas": "法玛斯",
             "usp_silencer": "USP-S", "glock": "格洛克", "hkp2000": "P2000", "p250": "P250", "tec9": "Tec-9", "fiveseven": "FN57", "mp9": "MP9",
             "mac10": "MAC-10", "aug": "AUG", "sg556": "SG553", "ssg08": "鸟狙", "hegrenade": "手雷", "inferno": "燃烧弹", "knife": "刀",
             "elite": "双枪", "cz75a": "CZ75", "ump45": "UMP", "mp7": "MP7", "p90": "P90", "xm1014": "连喷", "mag7": "MAG-7", "nova": "新星",
             "planted_c4": "C4 爆炸", "world": "摔落", "worldent": "摔落", "mp5sd": "MP5", "taser": "电击枪", "usp_silencer_off": "USP-S",
             "revolver": "R8", "smokegrenade": "烟雾弹", "flashbang": "闪光弹", "decoy": "诱饵弹", "molotov": "燃烧弹", "incgrenade": "燃烧弹",
             "bizon": "野牛", "g3sg1": "连狙", "scar20": "连狙", "m249": "M249", "negev": "内格夫", "sawedoff": "截短"}


def weapon_cn(w):
    w = str(w)
    if w.startswith("knife") or w == "bayonet":
        return "刀"
    return WEAPON_CN.get(w, w)
REASON_CN = {"t_killed": "全歼 T", "ct_killed": "全歼 CT", "bomb_exploded": "炸弹爆炸", "bomb_defused": "拆除炸弹", "time_ran_out": "时间耗尽"}
TEAM_CN = {"natus-vincere": "NAVI", "the-mongolz": "The MongolZ", "g2": "G2", "faze": "FaZe", "mouz": "MOUZ", "vitality": "Vitality", "spirit": "Spirit",
           "virtus-pro": "Virtus.pro", "heroic": "Heroic", "liquid": "Liquid", "furia": "FURIA", "falcons": "Falcons", "astralis": "Astralis",
           "complexity": "Complexity", "ence": "ENCE", "big": "BIG", "nip": "NIP", "ninjas-in-pyjamas": "NIP", "3dmax": "3DMAX", "aurora": "Aurora",
           "eternal-fire": "Eternal Fire", "pain": "paiN", "mibr": "MIBR", "gamerlegion": "GamerLegion", "tyloo": "TYLOO", "betboom": "BetBoom"}
MAP_CN = {m: m.replace("de_", "").capitalize() for m in em.MAPS}
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def team_name(slug):
    if not slug:
        return "自选阵容"
    return TEAM_CN.get(slug, slug.replace("-", " ").title())


# ======================================================================== 资源
class Assets:
    def __init__(self):
        self.G = em.Generator.load(paths.MODELS + "/event_team")
        self.M = mo.Motion.load()
        self.S4 = p4.Simulator.load()            # 用 P4 回合模型引导每回合胜负（更准），事件模型负责生成过程
        self.V = vt.Veto.load() if os.path.exists(f"{vt.MDIR}/veto_model.pt") else None     # Ban/Pick 模型（没训练时为 None）
        self.P = ph.Physics.load() if os.path.exists(f"{ph.MDIR}/grids.npz") else None        # 地图物理：视线、可走区域、真实对枪位置
        self.G.phys = self.P
        self.U = None                                                                        # 道具模型（深度学习，没训练时为 None）
        try:
            import p15_utility_model as um
            if os.path.exists(f"{um.MDIR}/utility_model.pt"):
                self.U = um.UtilityGen.load()
        except Exception as e:                                                               # 道具模型缺失或版本不符时照常出片，只是没有道具
            print("道具模型未加载：", e, file=sys.stderr)
        self.maps = pd.read_parquet(f"{C}/maps.parquet")
        pl = pd.read_parquet(f"{C}/players.parquet")
        self.name = dict(zip(pl.steamid.astype(str), pl.name))
        ro = pd.read_parquet(f"{C}/player_roles.parquet")
        self.role = {(str(s), int(y)): r for s, y, r in zip(ro.steamid, ro.year, ro.role)}
        py = pd.read_parquet(f"{C}/player_years.parquet")
        self.by_name = {}
        for s, n, y, r in zip(py.steamid.astype(str), py.name, py.year, py.rounds):
            k = (str(n).lower(), int(y))
            if r > self.by_name.get(k, (None, -1))[1]:
                self.by_name[k] = (s, r)
        cen = pd.read_parquet(f"{C}/place_centers.parquet")
        self.sites = {m: {p[-1]: (x, y) for p, x, y in zip(g.place, g.X, g.Y) if p in ("BombsiteA", "BombsiteB")} for m, g in cen.groupby("map_name")}

    def roster(self, slug, year):
        m = self.maps[((self.maps.team1_slug == slug) | (self.maps.team2_slug == slug)) & (self.maps.year == year)]
        ros = np.where(m.team1_slug == slug, m.roster1, m.roster2)
        c = Counter(s for r in ros for s in str(r).split(","))
        return [(s, int(year)) for s, _ in c.most_common(5)]

    def parse_players(self, spec):
        out = []
        for item in spec.split(","):
            n, y = item.rsplit(":", 1)
            s = self.by_name.get((n.strip().lower(), int(y)))
            if not s:
                raise SystemExit(f"找不到选手 {n} 在 {y} 年的数据")
            out.append((s[0], int(y)))
        return out


# ======================================================================== 剧本 → 走位
def plan_tensors(A, story, r, team_ids):
    """把剧本中的一个回合转成走位模型的条件张量。"""
    G, M = A.G, A.M
    roster = [(p["steamid"], p["year"]) for p in story["players"]]
    mi = em.MAPS.index(story["map"])
    a_ct = r["a_side"] == "CT"
    side = np.array([int(a_ct)] * 5 + [int(not a_ct)] * 5)
    sa, sb = r["score_before"]
    b = {"m": [mi], "rctx": p4.ctx_features(np.array([sa], float), np.array([sb], float), np.array([r["round"]]), np.array([float(a_ct)], np.float32)),
         "tm": [[M.V.itm(t[0]) if t else 0 for t in team_ids]], "tmy": [[M.V.itmy(t[0], t[1]) if t else 0 for t in team_ids]],
         "p": [[M.V.ip(s) for s, _ in roster]], "py": [[M.V.ipy(s, y) for s, y in roster]], "side": [side], "eq": [np.array(r["equip"]) / 1000.0]}
    E = {k: np.zeros((1, em.MAXEV), np.int64) for k in ("type", "t1", "v", "a", "vpl", "site")}
    xy = {k: np.zeros((1, em.MAXEV, 2), np.float32) for k in ("vxy", "axy")}
    E["type"][0, 0] = 1
    E["v"][:], E["a"][:] = 10, 10
    death = [None] * 10
    for j, e in enumerate(r["events"][:em.MAXEV - 1], start=1):
        E["t1"][0, j] = min(int(e["time"]), mo.K - 1)
        if e["type"] == "kill":
            E["type"][0, j], E["v"][0, j], E["a"][0, j] = 2, e["victim"], e["attacker"]
            E["vpl"][0, j] = M.V.ipl(e["victim_place"])
            for key, f in (("vxy", "victim_xy"), ("axy", "attacker_xy")):
                if e.get(f):
                    xy[key][0, j] = M.norm(mi, e[f][0], e[f][1])
            death[e["victim"]] = int(e["time"]) + 1
        elif e["type"] == "plant":
            E["type"][0, j], E["a"][0, j], E["site"][0, j] = 3, e["player"], 1 if e["site"] == "A" else 2
            if e.get("xy"):
                xy["vxy"][0, j] = M.norm(mi, e["xy"][0], e["xy"][1])
        else:
            E["type"][0, j] = 4
    def tens(v):                       # Apple GPU（MPS）不支持 float64，统一转 float32
        a = np.asarray(v)
        return torch.as_tensor(a.astype(np.float32) if a.dtype == np.float64 else a, device=mo.DEV)
    cond = {k: tens(v) for k, v in {**b, **E, **xy}.items()}
    cond["rctx"] = cond["rctx"].float()
    cond["eq"] = cond["eq"].float()
    n = int(min(mo.K, (r["events"][-1]["time"] if r["events"] else 60) + 3))
    return cond, death, n


def pick_key_rounds(story, live, k=3):
    rounds = story["rounds"]
    keys = {1}
    clutch = [r["round"] for r in rounds if r["events"] and r["events"][-1].get("clutch")]
    if clutch:
        keys.add(clutch[0])
    swing, best = None, -1
    for r in rounds:
        a, b = r["score_before"]
        p0 = live.get(f"{a}-{b}")
        a2, b2 = (a + 1, b) if r["winner"] == "A" else (a, b + 1)
        p1 = live.get(f"{a2}-{b2}")
        if p0 is not None and p1 is not None and abs(p1 - p0) > best and r["round"] not in keys:
            swing, best = r["round"], abs(p1 - p0)
    if swing:
        keys.add(swing)
    keys.add(rounds[-1]["round"])
    return sorted(keys)[:k + 1]


# ======================================================================== 渲染
class UtilFX:
    """道具画面：出手后沿彩色虚线飞出 → 烟雾云团（白，带倒计时）/ 火团（红橙闪烁，带倒计时）/ 闪光（黄色星形爆闪）/ 手雷（绿色爆炸）。"""
    COL = {"smoke": "#e9edf2", "molotov": "#ff4b2b", "flash": "#fff27a", "he": "#7ee081"}
    CLOUD = [(0, 0, 0.62), (0.42, 0.18, 0.5), (-0.4, 0.2, 0.5), (0.22, -0.38, 0.5), (-0.25, -0.35, 0.48)]       # 云团：几个圆叠成
    FLAME = [(0, 0, 0.55), (0.38, 0.15, 0.45), (-0.36, 0.12, 0.45), (0.1, -0.38, 0.42), (-0.15, 0.4, 0.4)]

    def __init__(self, ax, utils, tr):
        self.ax, self.tr = ax, tr
        self.u = sorted(utils or [], key=lambda x: x["t_throw"])
        self.art = {}

    def _p0(self, u):
        k = min(max(int(u["t_throw"]), 0), len(self.tr) - 1)
        return self.tr[k, u["slot"], :2]

    def smokes(self, t):
        return [(u["x"], u["y"]) for u in self.u if u["type"] == "smoke" and u["t_start"] <= t <= u["t_end"]]

    def fires(self, t):
        return [(u["x"], u["y"]) for u in self.u if u["type"] == "molotov" and u["t_start"] <= t <= u["t_end"]]

    def _drop(self, i):
        a = self.art.pop(i, {})
        seen = set()
        for v in a.values():
            for x in (v if isinstance(v, list) else [v]):
                if id(x) not in seen:
                    seen.add(id(x))
                    x.remove()

    def _make(self, u):
        c = self.COL[u["type"]]
        a = {"line": self.ax.plot([], [], ls=(0, (3, 2)), color=c, lw=2.2, alpha=0.95, zorder=3.6)[0],
             "dot": self.ax.scatter([], [], s=70, c=c, edgecolors="#111", linewidths=0.8, zorder=3.7)}
        if u["type"] == "smoke":
            a["parts"] = [Circle((u["x"] + dx * ph.SMOKE_R, u["y"] + dy * ph.SMOKE_R), r * ph.SMOKE_R, fc="#eef1f5", ec="#aeb8c4", lw=1.2,
                                 alpha=0.0, zorder=4.4) for dx, dy, r in self.CLOUD]
            a["timer"] = self.ax.text(u["x"], u["y"], "", color="#2b3442", fontsize=16, fontweight="bold", ha="center", va="center", zorder=4.6)
        elif u["type"] == "molotov":
            a["parts"] = [Circle((u["x"] + dx * ph.FIRE_R, u["y"] + dy * ph.FIRE_R), r * ph.FIRE_R, fc=fc, ec="none", alpha=0.0, zorder=3.4)
                          for (dx, dy, r), fc in zip(self.FLAME, ["#ff3b1f", "#ff7a1a", "#ff5a1f", "#ffb02e", "#ff3b1f"])]
            a["timer"] = self.ax.text(u["x"], u["y"], "", color="#ffffff", fontsize=15, fontweight="bold", ha="center", va="center", zorder=3.6)
        else:
            mk = "*" if u["type"] == "flash" else (8, 1, 0)
            a["burst"] = self.ax.scatter([u["x"]], [u["y"]], marker=mk, s=1800, c=c, edgecolors="#ffffff" if u["type"] == "he" else "#ffb300",
                                         linewidths=1.5, alpha=0.0, zorder=7.5)
            a["ring"] = Circle((u["x"], u["y"]), 60, fc="none", ec=c, lw=4, alpha=0.0, zorder=7.4)
            a["parts"] = [a["ring"]]
            a["timer"] = self.ax.text(u["x"], u["y"] - 140, "闪" if u["type"] == "flash" else "雷", color=c, fontsize=15, fontweight="bold",
                                      ha="center", va="top", alpha=0.0, zorder=7.6)
        for p in a["parts"]:
            self.ax.add_patch(p)
        return a

    def update(self, t):
        for i, u in enumerate(self.u):
            if t < u["t_throw"] or t > u["t_end"] + 0.8:
                if i in self.art:
                    self._drop(i)
                continue
            if i not in self.art:
                self.art[i] = self._make(u)
            a = self.art[i]
            if t < u["t_start"]:                                              # 飞行中
                f = (t - u["t_throw"]) / max(u["t_start"] - u["t_throw"], 0.05)
                p0 = self._p0(u)
                p = p0 + f * (np.array([u["x"], u["y"]]) - p0)
                a["line"].set_data([p0[0], p[0]], [p0[1], p[1]])
                a["dot"].set_offsets([p])
                continue
            a["line"].set_data([], [])
            a["dot"].set_offsets(np.zeros((0, 2)))
            age, left = t - u["t_start"], u["t_end"] - t
            if u["type"] == "smoke":
                grow = min(1.0, 0.35 + age / 1.0)
                al = 0.92 * min(1.0, max(0.0, left / 2.0)) if left > 0 else 0.0
                for (dx, dy, r), p in zip(self.CLOUD, a["parts"]):
                    p.set_radius(r * ph.SMOKE_R * grow)
                    p.set_alpha(al)
                a["timer"].set_text(f"{left:.0f}" if left > 0.5 else "")
            elif u["type"] == "molotov":
                al = min(1.0, max(0.0, left / 1.0)) if left > 0 else 0.0
                for j, ((dx, dy, r), p) in enumerate(zip(self.FLAME, a["parts"])):
                    p.set_radius(r * ph.FIRE_R * min(1.0, 0.4 + age) * (1 + 0.12 * np.sin(age * 11 + j * 1.7)))
                    p.set_alpha((0.62 + 0.18 * np.sin(age * 9 + j)) * al)
                a["timer"].set_text(f"{left:.0f}" if left > 0.5 else "")
            else:                                                             # 闪光 / 手雷：0.8 秒的爆闪
                f = min(1.0, age / 0.8)
                a["burst"].set_alpha(0.95 * (1 - f))
                a["burst"].set_sizes([1800 + 1500 * f])
                a["ring"].set_radius(60 + (300 if u["type"] == "flash" else 220) * f)
                a["ring"].set_alpha(0.85 * (1 - f))
                a["timer"].set_alpha(1 - f)

    def clear(self):
        for i in list(self.art):
            self._drop(i)


class Renderer:
    def __init__(self, A, story, res, meta, tracks, emit):
        self.A, self.s, self.res, self.meta, self.tracks = A, story, res, meta, tracks
        self.emit = emit                  # emit(帧, 重复次数)：边渲染边写入视频，避免占用大量内存
        self.cover = None
        self.map = story["map"]
        mi = em.MAPS.index(self.map)
        self.b = A.M.bounds[mi]
        dens = A.M.walk.get(self.map + "_density")
        self.fig = plt.figure(figsize=(10.8, 19.2), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0.0, 0.30, 1.0, 0.5625], facecolor=BG)
        self.ax.set_xlim(self.b[0] - 150, self.b[1] + 150)
        self.ax.set_ylim(self.b[2] - 150, self.b[3] + 150)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")
        P_ = getattr(A, "P", None)
        if P_ is not None and P_.has(self.map) and f"{self.map}_height" in P_.g:
            self.draw_terrain(P_)
        elif dens is not None:
            from matplotlib.colors import LinearSegmentedColormap
            from scipy.ndimage import gaussian_filter
            raw = np.log1p(dens.T.astype(float))
            walk = gaussian_filter((dens.T >= 3).astype(float), 0.8) > 0.35
            img = np.ma.masked_where(~walk, gaussian_filter(raw, 1.0))
            cmap = LinearSegmentedColormap.from_list("cs2", ["#1f2a3a", "#33465f", "#6a809e"])
            self.ax.imshow(img, origin="lower", cmap=cmap, interpolation="bilinear", vmin=0, vmax=np.percentile(img.compressed(), 98),
                           extent=[self.b[0], self.b[0] + dens.shape[0] * mo.WALK_CELL, self.b[2], self.b[2] + dens.shape[1] * mo.WALK_CELL], zorder=0)
        for site, (x, y) in A.sites.get(self.map, {}).items():
            if getattr(self, "terrain", False):                              # 导播风格：暗红底的包点字母
                self.ax.text(x, y, site, color="#ffffff", fontsize=30, ha="center", va="center", fontweight="bold", zorder=1,
                             bbox=dict(boxstyle="round,pad=0.25", fc="#7d2236", ec="#c9d1dc", lw=1.0, alpha=0.9))
            else:
                self.ax.text(x, y, site, color="#ffffff", alpha=0.25, fontsize=70, ha="center", va="center", fontweight="bold", zorder=1)
        if getattr(self, "terrain", False):
            self.legend()
        self.names = [p["name"] for p in story["players"]]
        self.team_of = ["A"] * 5 + ["B"] * 5
        self.dots = self.ax.scatter(np.zeros(10), np.zeros(10), s=560, c=[COL[t] for t in self.team_of], edgecolors="white", linewidths=1.8, zorder=5)
        self.dead = self.ax.scatter([], [], marker="x", s=220, c=[], linewidths=3, zorder=4)
        self.labels = [self.ax.text(0, 0, n, color=COL[self.team_of[i]], fontsize=13, fontweight="bold", ha="center", va="bottom", zorder=6) for i, n in enumerate(self.names)]
        self.nums = [self.ax.text(0, 0, "1234567890"[i], color="white", fontsize=15, fontweight="bold", ha="center", va="center", zorder=5.6) for i in range(10)]
        self.status = [self.ax.text(0, 0, "", color="white", fontsize=12, fontweight="bold", ha="center", va="top", zorder=6) for i in range(10)]
        self.yaw = [self.ax.plot([], [], color=COL[self.team_of[i]], lw=5, solid_capstyle="round", zorder=4.9)[0] for i in range(10)]
        self.halo = self.ax.scatter([], [], s=1500, c="#ffffff", alpha=0.0, zorder=4.8)                         # 被闪光
        self.burn = self.ax.scatter([], [], s=1000, facecolors="none", edgecolors="#ff4b2b", linewidths=3, zorder=5.5)  # 燃烧中
        self.t_bomb = self.ax.text(0, 0, "", color="#ff6b6b", fontsize=16, fontweight="bold", ha="center", va="bottom", zorder=8)
        self.kill_lines = []
        self.bomb_mark = self.ax.scatter([], [], marker="*", s=700, c="#ff4d4f", zorder=7)
        self.summary_marks = self.ax.scatter([], [], marker="x", s=160, c=[], linewidths=2.5, zorder=4)
        # 文字区
        f = self.fig
        f.text(0.5, 0.982, "AI 模拟对局 · 非真实比赛", color="#ff6b6b", fontsize=20, ha="center", va="top", alpha=0.9)
        f.text(0.5, 0.955, f"{meta['title']}", color=DIM if not meta.get("series_bar") else "#dfe6ee", fontsize=22 if len(meta["title"]) <= 28 else 18,
               ha="center", va="top")
        fs = lambda name: 34 if len(name) <= 13 else max(20, int(34 * 13 / len(name)))       # 长队名自动缩小，避免压到比分
        self.t_name_a = f.text(0.04, 0.905, meta["a_name"], color=COL["A"], fontsize=fs(meta["a_name"]), fontweight="bold", ha="left", va="center")
        self.t_name_b = f.text(0.96, 0.905, meta["b_name"], color=COL["B"], fontsize=fs(meta["b_name"]), fontweight="bold", ha="right", va="center")
        self.t_score = f.text(0.5, 0.905, "0 : 0", color=TXT, fontsize=64, fontweight="bold", ha="center", va="center")
        self.t_round = f.text(0.5, 0.872, "", color=DIM, fontsize=22, ha="center", va="center")
        self.t_clock = f.text(0.94, 0.872, "", color=DIM, fontsize=22, ha="right", va="center")
        self.t_card = f.text(0.5, 0.58, "", color=TXT, fontsize=40, fontweight="bold", ha="center", va="center", zorder=10,
                             bbox=dict(boxstyle="round,pad=0.6", fc=PANEL, ec="#2b3240", alpha=0.0))
        self.feed = [f.text(0.05, 0.292 - i * 0.021, "", color=TXT, fontsize=21, ha="left", va="top") for i in range(5)]
        self.t_eco = f.text(0.5, 0.171, "", color=DIM, fontsize=19, ha="center", va="center")
        self.wp_ax = f.add_axes([0.2, 0.03, 0.74, 0.115], facecolor=PANEL)
        self.wp_ax.set_ylim(0, 1)
        self.wp_ax.set_xlim(0, max(len(story["rounds"]), 24) + 1)
        self.wp_ax.axhline(0.5, color="#3a4250", lw=1)
        for sp in self.wp_ax.spines.values():
            sp.set_visible(False)
        self.wp_ax.tick_params(colors=DIM, labelsize=12)
        self.wp_ax.set_yticks([0, 0.5, 1])
        self.wp_ax.set_yticklabels([meta["b_short"], "50%", meta["a_short"]], fontsize=15)
        for lab, c in zip(self.wp_ax.get_yticklabels(), (COL["B"], DIM, COL["A"])):
            lab.set_color(c)
        self.wp_line, = self.wp_ax.plot([], [], color=COL["A"], lw=3)
        self.wp_ax.set_title("实时胜率（模拟推演）", color=DIM, fontsize=15, loc="left")
        self.overlay = []
        self.wp_hist = [(0, res["p_win_A"])]

    def draw_terrain(self, P_):
        """导播风格底图：深蓝地板（越亮越高）、浅灰墙线、浅灰小障碍物（箱子、柱子）、暗红包点区域。全部来自真实站位数据。"""
        from scipy.ndimage import distance_transform_edt, gaussian_filter, label
        m = self.map
        walk = P_.g[f"{m}_walk"]
        h = P_.g[f"{m}_height"].astype(float)
        site = P_.g.get(f"{m}_site", np.zeros_like(walk, np.int8))
        o = P_.meta["origin"][m]
        cell = ph.CELL
        nx, ny = walk.shape
        mf = gaussian_filter(walk.astype(float), 0.7)
        mask = mf > 0.45
        lab, n = label(~mask)                                                   # 不可走区域里，不连到地图外面的小块 = 箱子 / 柱子
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = [i for i in range(1, n + 1) if i not in border and sizes[i] <= 160]
        box = np.isin(lab, small)
        valid = ~np.isnan(h)
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        hs = gaussian_filter(h[tuple(idx)], 1.0)
        lo, hi = np.percentile(hs[mask], [2, 98])
        hn = np.clip((hs - lo) / max(hi - lo, 1.0), 0, 1)
        dzdx, dzdy = np.gradient(hs, cell)
        shade = np.clip(1.0 - 0.9 * (dzdx * 0.7 + dzdy * 0.7), 0.82, 1.12)        # 很轻的坡面明暗，看得出台阶和坡道
        base, high = np.array([0.12, 0.16, 0.34]), np.array([0.25, 0.32, 0.60])
        rgb = (base + (high - base) * hn[..., None]) * shade[..., None]
        sm = (site > 0) & mask
        rgb[sm] = rgb[sm] * 0.35 + np.array([0.46, 0.12, 0.20]) * 0.65            # 包点区域
        alpha = mask.astype(float)
        rgb[box] = np.array([0.60, 0.65, 0.72])
        alpha[box] = 1.0
        rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1)
        ext = [o[0], o[0] + nx * cell, o[1], o[1] + ny * cell]
        self.ax.imshow(rgba.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="bilinear", zorder=0)
        xs = o[0] + (np.arange(nx) + 0.5) * cell
        ys = o[1] + (np.arange(ny) + 0.5) * cell
        self.ax.contour(xs, ys, mf.T, levels=[0.45], colors="#c9d1dc", linewidths=1.7, zorder=0.8)      # 墙线
        if box.any():
            self.ax.contour(xs, ys, gaussian_filter(box.astype(float), 0.5).T, levels=[0.5], colors="#e8edf3", linewidths=1.0, zorder=0.9)
        self.terrain = True

    def utility_effects(self, r, tr, fx, events):
        """按游戏机制近似算道具效果：
        闪光——爆点与选手之间有视线（不被烟挡）、距离 1600 以内；朝向偏差 ≤53° 全致盲，越偏越短；距离越远越短；最长约 4.8 秒。
        手雷——350 单位内有视线的选手扣血（有甲时最多约 57 点，随距离线性衰减）。
        返回 (每人的致盲区间 {位置号: [(开始, 结束)]}, 扣血飘字列表 [(文字对象, 时间, 位置号)])。"""
        blind, dmg = {}, []
        P_ = self.A.P
        death = {e["victim"]: e["time"] for e in events if e["type"] == "kill"}
        n = len(tr)
        for u in fx.u:
            if u["type"] not in ("flash", "he"):
                continue
            t0 = u["t_start"]
            k = min(max(int(t0), 0), n - 1)
            k2 = min(k + 1, n - 1)
            fr = t0 - int(t0)
            for i in range(10):
                if death.get(i, 1e9) <= t0:
                    continue
                p = tr[k, i, :2] * (1 - fr) + tr[k2, i, :2] * fr
                d = float(np.hypot(p[0] - u["x"], p[1] - u["y"]))
                if d > 1600:
                    continue
                if P_ is not None and not P_.los(self.map, p[0], p[1], u["x"], u["y"])[0]:
                    continue
                sm = fx.smokes(t0)
                if sm and ph.seg_hits_circles(p[0], p[1], u["x"], u["y"], sm, ph.SMOKE_R):
                    continue
                if u["type"] == "flash":
                    ang = np.degrees(np.arctan2(u["y"] - p[1], u["x"] - p[0]))
                    off = abs((ang - tr[k, i, 2] + 180) % 360 - 180)
                    fa = 1.0 if off <= 53 else (0.6 if off <= 72 else (0.35 if off <= 101 else 0.1))
                    fd = float(np.clip(1 - (d - 300) / 1300, 0, 1))
                    dur = 4.8 * fa * fd
                    if dur >= 0.5:
                        blind.setdefault(i, []).append((t0, t0 + dur))
                elif d < 350:
                    hp = int(round(57 * (1 - d / 350)))
                    if hp >= 5:
                        txt = self.ax.text(0, 0, f"-{hp}", color="#ff5252", fontsize=17, fontweight="bold", ha="left", va="bottom", zorder=8, visible=False)
                        dmg.append([txt, t0, i])
        return blind, dmg

    def legend(self):
        """右上角的道具图例。"""
        items = [("●", "烟", UtilFX.COL["smoke"]), ("●", "火", UtilFX.COL["molotov"]), ("★", "闪", UtilFX.COL["flash"]), ("◆", "雷", UtilFX.COL["he"])]
        for i, (g, t, c) in enumerate(items):
            x = 0.62 + i * 0.095
            self.ax.text(x, 0.985, g, color=c, fontsize=16, transform=self.ax.transAxes, ha="right", va="top", zorder=9)
            self.ax.text(x + 0.005, 0.985, t, color="#dfe6ee", fontsize=14, transform=self.ax.transAxes, ha="left", va="top", zorder=9)
        self.ax.text(0.02, 0.985, "地板越亮越高", color=DIM, fontsize=13, transform=self.ax.transAxes, ha="left", va="top", zorder=9)

    # ---- 通用
    def frame(self):
        for t in self.ax.texts:                       # 地图坐标里的文字（名字、编号、状态、倒计时）裁剪在地图框内；高光镜头放大时不会漏到框外
            if t.get_transform() is self.ax.transData and not t.get_clip_on():
                t.set_clip_on(True)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()

    def clear_overlay(self):
        for o in self.overlay:
            o.remove()
        self.overlay = []

    def set_score(self, a, b, rnd, side_a):
        self.t_score.set_text(f"{a} : {b}")
        self.t_round.set_text(f"第 {rnd} 回合 · {self.meta['a_name']} {'CT' if side_a == 'CT' else 'T'} 方")

    def show_players(self, visible):
        self.dots.set_visible(visible)
        for t in self.labels + self.yaw + self.nums + self.status:
            t.set_visible(visible)

    def set_feed(self, kills):
        for i, t in enumerate(self.feed):
            if i < len(kills):
                e = kills[-1 - i]
                hs = " ⊕爆头" if e["headshot"] else ""
                t.set_text(f"{e['attacker_name']}  [{weapon_cn(e['weapon'])}]  →  {e['victim_name']}{hs}{'  穿墙' if e.get('wallbang') else ''}{'  穿烟' if e.get('thrusmoke') else ''}{'  (被闪)' if e.get('victim_blind') else ''}")
                t.set_color(COL["A" if e["attacker"] < 5 else "B"])
            else:
                t.set_text("")

    def push_wp(self, rnd, p):
        self.wp_hist.append((rnd, p))
        xs, ys = zip(*self.wp_hist)
        self.wp_line.set_data(xs, ys)

    # ---- 片头 / 片尾
    def intro(self):
        self.t_eco.set_text("")
        self.show_players(False)
        p = self.res["p_win_A"]
        lines = []
        f = self.fig
        for side, x, ha in (("A", 0.08, "left"), ("B", 0.92, "right")):
            ps = [q for q in self.s["players"] if q["team"] == side]
            for i, q in enumerate(ps):
                role = self.A.role.get((q["steamid"], q["year"]), "")
                lines.append(f.text(x, 0.74 - i * 0.045, f"{q['name']}", color=COL[side], fontsize=30, fontweight="bold", ha=ha, va="center"))
                lines.append(f.text(x, 0.72 - i * 0.045, f"{q['year']} · {role}", color=DIM, fontsize=17, ha=ha, va="center"))
        lines.append(f.text(0.5, 0.79, "VS", color=TXT, fontsize=60, fontweight="bold", ha="center", va="center"))
        lines.append(f.text(0.5, 0.47, f"AI 赛前预测（模拟 {self.meta['n_sims']} 场）", color=DIM, fontsize=22, ha="center"))
        bar = self.fig.add_axes([0.1, 0.42, 0.8, 0.035])
        bar.barh([0], [p], color=COL["A"])
        bar.barh([0], [1 - p], left=[p], color=COL["B"])
        bar.set_xlim(0, 1)
        bar.axis("off")
        lines.append(f.text(0.1, 0.40, f"{p:.0%}", color=COL["A"], fontsize=34, fontweight="bold", ha="left", va="top"))
        lines.append(f.text(0.9, 0.40, f"{1 - p:.0%}", color=COL["B"], fontsize=34, fontweight="bold", ha="right", va="top"))
        lines.append(f.text(0.5, 0.36, f"地图：{MAP_CN[self.map]}", color=TXT, fontsize=26, ha="center", va="top"))
        self.ax.set_visible(False)
        img = self.frame()
        self.cover = img
        self.emit(img, INTRO_FRAMES)
        for t in lines:
            t.remove()
        bar.remove()
        self.ax.set_visible(True)

    def outro(self):
        self.t_eco.set_text("")
        self.show_players(False)
        self.clear_overlay()
        fa, fb = self.s["final"]
        win = "A" if fa > fb else "B"
        f = self.fig
        items = [f.text(0.5, 0.78, f"{self.meta['a_name' if win == 'A' else 'b_name']} 获胜", color=COL[win], fontsize=48, fontweight="bold", ha="center")]
        self.ax.set_visible(False)
        y0 = 0.72
        items.append(f.text(0.5, y0, "选手    击杀 - 死亡    首杀    残局", color=DIM, fontsize=20, ha="center"))
        rows = []
        for i, q in enumerate(sorted(self.s["players"], key=lambda q: (q["team"], -q["kills"]))):
            rows.append((f.text(0.5, y0 - 0.035 * (i + 1) - (0.02 if q["team"] == "B" else 0),
                                f"{q['name']:<12} {q['kills']:>3} - {q['deaths']:<3}   {q['opening_kills']:>2}    {q['clutch_wins']:>2}",
                                color=COL[q["team"]], fontsize=22, ha="left", family="monospace"), q))
        rend = self.canvas.get_renderer()                   # 各行左对齐、整体居中；MVP 标记单独画在行尾，不再把那一行挤歪
        w = max(t.get_window_extent(rend).width for t, _ in rows) / f.bbox.width
        for t, q in rows:
            t.set_x(0.5 - w / 2)
            items.append(t)
            if q["slot"] == self.s["mvp"]:
                items.append(f.text(0.5 + w / 2 + 0.012, t.get_position()[1], "★ MVP", color="#ffd666", fontsize=20, fontweight="bold", ha="left"))
        items.append(f.text(0.5, 0.31, f"赛前模拟 {self.meta['n_sims']} 场：{self.meta['a_name']} 胜率 {self.res['p_win_A']:.0%}", color=TXT, fontsize=24, ha="center"))
        top = sorted(self.res["scores"].items(), key=lambda x: -x[1])[:3]
        items.append(f.text(0.5, 0.285, "最可能比分：" + "  ".join(f"{k}（{v:.0%}）" for k, v in top), color=DIM, fontsize=20, ha="center"))
        if self.meta.get("series_after"):
            items.append(f.text(0.5, 0.215, self.meta["series_after"], color="#ffd666", fontsize=34, fontweight="bold", ha="center"))
        for t in self.feed:
            t.set_text("")
        self.emit(self.frame(), OUTRO_FRAMES)
        for t in items:
            t.remove()
        self.ax.set_visible(True)

    # ---- 选手高光：片头 / 片尾
    def highlight_card(self, q, lines, frames, big="选手高光"):
        self.t_eco.set_text("")
        self.show_players(False)
        self.clear_overlay()
        self.ax.set_visible(False)
        f = self.fig
        side = q["team"]
        role = self.A.role.get((q["steamid"], q["year"]), "")
        items = [f.text(0.5, 0.80, big, color="#ffd666", fontsize=40, fontweight="bold", ha="center"),
                 f.text(0.5, 0.71, q["name"], color=COL[side], fontsize=76, fontweight="bold", ha="center"),
                 f.text(0.5, 0.665, f"{self.meta['a_name' if side == 'A' else 'b_name']} · {q['year']}" + (f" · {role}" if role else ""),
                        color=DIM, fontsize=24, ha="center")]
        for i, ln in enumerate(lines):
            items.append(f.text(0.5, 0.58 - i * 0.05, ln, color=TXT, fontsize=30, ha="center"))
        img = self.frame()
        self.emit(img, frames)
        for t in items:
            t.remove()
        self.ax.set_visible(True)
        return img

    # ---- 普通回合：快速带过
    def summary_round(self, r, score_after, live):
        self.show_players(False)
        self.clear_overlay()
        kills = [e for e in r["events"] if e["type"] == "kill"]
        xs = [e["victim_xy"] for e in kills if e.get("victim_xy")]
        cs = [COL["A" if e["victim"] < 5 else "B"] for e in kills if e.get("victim_xy")]
        self.summary_marks.set_offsets(np.array(xs) if xs else np.zeros((0, 2)))
        self.summary_marks.set_color(cs)
        end = r["events"][-1] if r["events"] else {}
        wname = self.meta["a_name"] if r["winner"] == "A" else self.meta["b_name"]
        txt = f"第 {r['round']} 回合  {wname} 赢\n{REASON_CN.get(end.get('reason'), '')}"
        if end.get("clutch"):
            txt += f"  ·  {end['clutch']['player_name']} 1v{end['clutch']['vs']} 残局！"
        self.t_card.set_text(txt)
        self.t_card.set_color(COL[r["winner"]] if r["winner"] else TXT)
        self.t_card.get_bbox_patch().set_alpha(0.85)
        self.set_feed(kills)
        self.set_score(*r["score_before"], r["round"], r["a_side"])
        eq = r["equip"]
        self.t_eco.set_text(f"装备价值  {self.meta['a_short']} \\${sum(eq[:5]):,}   vs   {self.meta['b_short']} \\${sum(eq[5:]):,}")
        self.emit(self.frame(), SUMMARY_FRAMES // 2)
        self.t_score.set_text(f"{score_after[0]} : {score_after[1]}")
        p = live.get(f"{score_after[0]}-{score_after[1]}")
        if p is not None:
            self.push_wp(r["round"], p)
        self.emit(self.frame(), SUMMARY_FRAMES - SUMMARY_FRAMES // 2)
        self.t_card.set_text("")
        self.t_card.get_bbox_patch().set_alpha(0.0)
        self.summary_marks.set_offsets(np.zeros((0, 2)))

    # ---- 关键回合：完整 2D 走位
    def key_round(self, r, tr, score_after, live, focus=None):
        """focus：选手高光时的位置号——镜头跟随并放大、金色光圈、他的击杀放慢。"""
        self.clear_overlay()
        self.show_players(True)
        self.set_score(*r["score_before"], r["round"], r["a_side"])
        eq = r["equip"]
        self.t_eco.set_text(f"装备价值  {self.meta['a_short']} \\${sum(eq[:5]):,}   vs   {self.meta['b_short']} \\${sum(eq[5:]):,}")
        ttl = "★ 关键回合 ★" if focus is None else f"★ {self.names[focus]} 高光 ★"
        title = self.fig.text(0.5, 0.835, ttl, color="#ffd666", fontsize=26, fontweight="bold", ha="center", va="center")
        self.overlay.append(title)
        n = len(tr)
        events = sorted(r["events"], key=lambda e: e["time"])
        shown, kill_vis, planted = [], [], None
        step = SPEED / FPS
        t = max(0.0, (events[0]["time"] - 12) if events else 0.0)
        t_end = min(n - 1.0, events[-1]["time"] + 1.5) if events else n - 1.0
        dead = {}
        ei = 0
        fx = UtilFX(self.ax, r.get("utility"), tr)
        blind, dmg_txt = self.utility_effects(r, tr, fx, events)
        bomb_t = None
        lim0 = (self.ax.get_xlim(), self.ax.get_ylim())
        ring = self.ax.scatter([], [], s=1100, facecolors="none", edgecolors="#ffd666", linewidths=3.5, zorder=6.5)
        focus_kills = [e["time"] for e in events if e["type"] == "kill" and e["attacker"] == focus] if focus is not None else []
        cam = None
        while t <= t_end:
            k = int(t)
            fr = t - k
            k2 = min(k + 1, n - 1)
            pos = tr[k, :, :2] * (1 - fr) + tr[k2, :, :2] * fr
            if focus is not None:                                              # 镜头跟随高光选手
                tgt = pos[focus] if focus not in dead else (cam if cam is not None else pos[focus])
                cam = tgt.copy() if cam is None else cam * 0.85 + tgt * 0.15
                half = 950.0
                self.ax.set_xlim(cam[0] - half, cam[0] + half)
                self.ax.set_ylim(cam[1] - half, cam[1] + half)
                ring.set_offsets([pos[focus]] if focus not in dead else np.zeros((0, 2)))
            while ei < len(events) and events[ei]["time"] <= t:
                e = events[ei]
                if e["type"] == "kill":
                    a, v = e["attacker"], e["victim"]
                    dead[v] = pos[v].copy()
                    if any(b0 <= e["time"] <= b1 for b0, b1 in blind.get(v, [])):
                        e["victim_blind"] = True                         # 被闪着的时候被杀
                    if ph.needs_los(e.get("weapon", "")) and self.A.P is not None and \
                            not self.A.P.los(self.map, pos[a][0], pos[a][1], pos[v][0], pos[v][1])[0]:
                        e["wallbang"] = True                           # 极少数时间线来不及拉出视线的击杀，如实标为穿墙
                    sm = fx.smokes(t)
                    if sm and ph.needs_los(e.get("weapon", "")) and ph.seg_hits_circles(pos[a][0], pos[a][1], pos[v][0], pos[v][1], sm, ph.SMOKE_R):
                        e["thrusmoke"] = True                          # 隔着烟打死（真实比赛约 9% 的击杀是穿烟）
                    if ph.needs_los(e.get("weapon", "")):              # 枪械击杀：画交火连线
                        line, = self.ax.plot([pos[a][0], pos[v][0]], [pos[a][1], pos[v][1]], color=COL["A" if a < 5 else "B"], lw=4, alpha=0.95, zorder=4)
                    else:                                              # 手雷/燃烧弹/炸弹：不需要视线，只在倒地处画爆炸标记
                        line = self.ax.scatter([pos[v][0]], [pos[v][1]], marker="*", s=900, c="#ff9c3a", alpha=0.95, zorder=4)
                    kill_vis.append([line, 12])
                    shown.append(e)
                    self.set_feed(shown)
                elif e["type"] == "plant":
                    planted = pos[e["player"]].copy()
                    bomb_t = e["time"]
                    self.bomb_mark.set_offsets([planted])
                    self.overlay.append(self.ax.text(planted[0], planted[1] + 120, f"{e['player_name']} 下包 {e['site']}", color="#ff7875",
                                                     fontsize=18, ha="center", zorder=8))
                elif e["type"] == "end":
                    wname = self.meta["a_name"] if r["winner"] == "A" else self.meta["b_name"]
                    txt = f"{wname} 赢下第 {r['round']} 回合\n{REASON_CN.get(e['reason'], '')}"
                    if e.get("clutch"):
                        txt += f"\n{e['clutch']['player_name']} 1v{e['clutch']['vs']} 残局！"
                    self.t_card.set_text(txt)
                    self.t_card.set_color(COL[r["winner"]])
                    self.t_card.get_bbox_patch().set_alpha(0.85)
                ei += 1
            alive = [i for i in range(10) if i not in dead]
            self.dots.set_offsets(pos[alive])
            self.dots.set_color([COL[self.team_of[i]] for i in alive])
            self.dead.set_offsets(np.array([dead[i] for i in dead]) if dead else np.zeros((0, 2)))
            self.dead.set_color([COL[self.team_of[i]] for i in dead])
            halo, halo_a, burning = [], [], []
            for i in range(10):
                if i in dead:
                    for a_ in (self.labels[i], self.nums[i], self.status[i]):
                        a_.set_visible(False)
                    self.yaw[i].set_data([], [])
                    continue
                self.labels[i].set_visible(True)
                self.labels[i].set_position((pos[i][0], pos[i][1] + 62))
                self.nums[i].set_visible(True)
                self.nums[i].set_position((pos[i][0], pos[i][1]))
                ang = np.deg2rad(tr[k, i, 2])
                self.yaw[i].set_data([pos[i][0] + 50 * np.cos(ang), pos[i][0] + 100 * np.cos(ang)], [pos[i][1] + 50 * np.sin(ang), pos[i][1] + 100 * np.sin(ang)])
                st_txt, st_col = "", "white"                                    # 状态：致盲 / 燃烧 / 烟中
                bl = max([b1 - t for b0, b1 in blind.get(i, []) if b0 <= t <= b1] or [0])
                if bl > 0:
                    st_txt, st_col = f"致盲 {bl:.1f}s", "#ffffff"
                    halo.append(pos[i])
                    halo_a.append(min(0.75, 0.25 + bl / 4))
                elif any(np.hypot(pos[i][0] - fx_, pos[i][1] - fy_) < ph.FIRE_R for fx_, fy_ in fx.fires(t)):
                    st_txt, st_col = "燃烧", "#ff6a3a"
                    burning.append(pos[i])
                elif any(np.hypot(pos[i][0] - sx_, pos[i][1] - sy_) < ph.SMOKE_R for sx_, sy_ in fx.smokes(t)):
                    st_txt, st_col = "烟中", "#c9d1dc"
                self.status[i].set_visible(bool(st_txt))
                self.status[i].set_text(st_txt)
                self.status[i].set_color(st_col)
                self.status[i].set_position((pos[i][0], pos[i][1] - 58))
            self.halo.set_offsets(np.array(halo) if halo else np.zeros((0, 2)))
            if halo:
                self.halo.set_alpha(max(halo_a))
            self.burn.set_offsets(np.array(burning) if burning else np.zeros((0, 2)))
            for hx in dmg_txt:                                                   # 手雷扣血：飘字 1.2 秒
                hx[0].set_visible(hx[1] <= t <= hx[1] + 1.2)
                if hx[1] <= t <= hx[1] + 1.2:
                    hx[0].set_position((pos[hx[2]][0] + 70, pos[hx[2]][1] + 40 + 120 * (t - hx[1])))
            if planted is not None and bomb_t is not None:                       # C4 倒计时（40 秒）
                left = 40.0 - (t - bomb_t)
                self.t_bomb.set_text(f"C4 {max(0.0, left):.0f}s" if left > 0 else "")
                self.t_bomb.set_position((planted[0], planted[1] + 60))
            for kv in kill_vis:
                kv[1] -= 1
                kv[0].set_alpha(max(0.0, kv[1] / 12))
            fx.update(t)
            self.t_clock.set_text(f"{int(t) // 60}:{int(t) % 60:02d}")
            self.emit(self.frame(), 1)
            t += step if not any(abs(t - kt) < 1.2 for kt in focus_kills) else step * 0.3       # 高光选手击杀前后放慢
        self.t_score.set_text(f"{score_after[0]} : {score_after[1]}")
        p = live.get(f"{score_after[0]}-{score_after[1]}")
        if p is not None:
            self.push_wp(r["round"], p)
        self.emit(self.frame(), 30)
        fx.clear()
        ring.remove()
        for hx in dmg_txt:
            hx[0].remove()
        self.halo.set_offsets(np.zeros((0, 2)))
        self.burn.set_offsets(np.zeros((0, 2)))
        self.t_bomb.set_text("")
        for a_ in self.status:
            a_.set_text("")
        self.ax.set_xlim(*lim0[0])
        self.ax.set_ylim(*lim0[1])
        for kv in kill_vis:
            kv[0].remove()
        self.bomb_mark.set_offsets(np.zeros((0, 2)))
        self.dead.set_offsets(np.zeros((0, 2)))
        self.t_card.set_text("")
        self.t_card.get_bbox_patch().set_alpha(0.0)
        self.t_clock.set_text("")


def team_meta(teamA, teamB, team_ids, map_name, n_sims):
    yrs = sorted({y for _, y in teamA + teamB})
    meta = {"a_name": f"{team_name(team_ids[0][0]) if team_ids[0] else '自选A'}" + (f" {team_ids[0][1]}" if team_ids[0] else ""),
            "b_name": f"{team_name(team_ids[1][0]) if team_ids[1] else '自选B'}" + (f" {team_ids[1][1]}" if team_ids[1] else ""),
            "title": f"梦幻对决 · {MAP_CN[map_name]} · " + "/".join(map(str, yrs)) if map_name else "梦幻对决 · " + "/".join(map(str, yrs)),
            "n_sims": n_sims,
            "a_short": team_name(team_ids[0][0]) if team_ids[0] else "自选A", "b_short": team_name(team_ids[1][0]) if team_ids[1] else "自选B"}
    if meta["a_short"] == meta["b_short"] and team_ids[0] and team_ids[1]:          # 同一战队不同年份：简称后加年份区分
        meta["a_short"] += f"'{str(team_ids[0][1])[-2:]}"
        meta["b_short"] += f"'{str(team_ids[1][1])[-2:]}"
    return meta


def sub_prog(prog, p0, p1):
    """把 0~1 的子进度映射到总进度的 [p0, p1] 区间。"""
    return lambda f, d: prog(p0 + (p1 - p0) * f, d)


def prepare_map(A, teamA, teamB, map_name, team_ids, n_sims, seed, story, key_k=3, winner=None, prog=None, log=print):
    """模拟整场 + 生成关键回合走位。返回 (res, 剧本, 关键回合, 走位)。进度 0~1。"""
    prog = prog or (lambda f, d: None)
    t0 = time.time()
    prog(0.0, f"模拟 {n_sims} 场比赛")
    res = A.G.generate_match(teamA, teamB, map_name, n_sims=n_sims, seed=seed, story=story, team_ids=team_ids, guide=A.S4, winner=winner,
                             progress=lambda rr: prog((0.6 * min(rr, 24) / 24 + 0.05 * min(max(rr - 24, 0), 12) / 12) / 0.72,
                                                      f"模拟 {n_sims} 场比赛：打到第 {rr} 回合" + ("（有几场进了加时）" if rr > 24 else "")))
    st = res["story"]
    log(f"  模拟 {n_sims} 场 {time.time() - t0:.0f}s：A 胜率 {res['p_win_A']:.1%}，剧本比分 {st['final']}")
    keys = pick_key_rounds(st, res["live_win_A"], k=key_k)
    tracks = {}
    for r in st["rounds"]:
        if r["round"] in keys:
            prog((0.66 + 0.06 * len(tracks) / max(1, len(keys))) / 0.72, f"生成关键回合走位 {len(tracks) + 1}/{len(keys)}")
            if A.U is not None:                                               # 道具：深度学习模型按剧本生成，落点取真实投掷记录
                r["utility"] = A.U.generate(st, r, team_ids, seed=seed + r["round"])
            cond, death, n = plan_tensors(A, st, r, team_ids)
            tracks[r["round"]] = A.M.generate(cond, map_name, n, death, seed=seed + r["round"], events=r["events"], phys=A.P,
                                              utility=r.get("utility"))
    log(f"  关键回合 {keys} 走位生成 {time.time() - t0:.0f}s")
    return res, st, keys, tracks


def render_map(A, res, st, tracks, meta, emit, prog=None):
    """把一张图渲染进同一个视频流（片头 → 各回合 → 片尾）。返回封面帧。进度 0~1。"""
    prog = prog or (lambda f, d: None)
    R = Renderer(A, st, res, meta, tracks, emit)
    prog(0.0, "渲染开场")
    R.intro()
    live = res["live_win_A"]
    a = b = 0
    weight = [30 if r["round"] in tracks else 1 for r in st["rounds"]]      # 关键回合逐帧画，比普通回合慢很多
    done, total, nr = 0, sum(weight), len(st["rounds"])
    for wi, r in zip(weight, st["rounds"]):
        prog(0.03 + 0.93 * done / total, f"渲染视频：第 {r['round']}/{nr} 回合" + ("（关键回合，较慢）" if wi > 1 else ""))
        done += wi
        a2, b2 = (a + 1, b) if r["winner"] == "A" else (a, b + 1)
        if r["round"] in tracks:
            R.key_round(r, tracks[r["round"]], (a2, b2), live)
        else:
            R.summary_round(r, (a2, b2), live)
        a, b = a2, b2
    prog(0.97, "渲染结尾")
    R.outro()
    plt.close(R.fig)
    return R.cover


def open_writer(out):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
    n_frames = [0]

    def emit(img, n):
        for _ in range(n):
            w.append_data(img)
        n_frames[0] += n
    return w, emit, n_frames


def make_video(A, teamA, teamB, map_name, out, team_ids=(None, None), n_sims=400, seed=0, story="typical", key_k=3, log=print,
               progress=None):
    """单张地图。progress(0~1 的进度, 说明文字)：可选，网页里显示进度条用。"""
    import imageio.v2 as imageio
    prog = progress or (lambda f, d: None)
    t0 = time.time()
    res, st, keys, tracks = prepare_map(A, teamA, teamB, map_name, team_ids, n_sims, seed, story, key_k, prog=sub_prog(prog, 0.01, 0.72), log=log)
    meta = team_meta(teamA, teamB, team_ids, map_name, n_sims)
    w, emit, n_frames = open_writer(out)
    with w:
        cover = render_map(A, res, st, tracks, meta, emit, sub_prog(prog, 0.72, 0.98))
    prog(0.99, "保存封面和剧本")
    imageio.imwrite(out.rsplit(".", 1)[0] + ".png", cover)
    st_out = {**{k: v for k, v in res.items() if k != "story"}, "story": st, "meta": meta, "key_rounds": keys}
    json.dump(st_out, open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1, default=float)
    log(f"  完成 {out}：{n_frames[0] / FPS:.0f} 秒视频，总用时 {time.time() - t0:.0f}s")
    return out


def make_highlight_video(A, teamA, teamB, map_name, out, focus, team_ids=(None, None), n_sims=400, seed=0, max_rounds=4, log=print, progress=None):
    """选手个人高光：从模拟里挑这名选手发挥最好的一场，只播他的高光回合（多杀 / 残局 / 首杀），镜头跟随、击杀慢放。
    focus: (steamid, 年份)，必须是两队 10 人之一。"""
    import imageio.v2 as imageio
    prog = progress or (lambda f, d: None)
    t0 = time.time()
    focus = (str(focus[0]), int(focus[1]))
    prog(0.01, f"模拟 {n_sims} 场，挑他发挥最好的一场")
    res = A.G.generate_match(teamA, teamB, map_name, n_sims=n_sims, seed=seed, story=("player", focus), team_ids=team_ids, guide=A.S4,
                             progress=lambda rr: prog(0.55 * min(rr, 24) / 24, f"模拟 {n_sims} 场比赛：打到第 {rr} 回合"))
    st = res["story"]
    q = next(p for p in st["players"] if p["steamid"] == focus[0] and p["year"] == focus[1])
    fs = q["slot"]
    scored = []
    for r in st["rounds"]:
        kills = [e for e in r["events"] if e["type"] == "kill" and e["attacker"] == fs]
        end = next((e for e in r["events"] if e["type"] == "end"), {})
        clutch = end.get("clutch", {}).get("player") == fs
        opening = any(e.get("opening") for e in kills)
        sc = len(kills) * 2 + (5 if clutch else 0) + (1 if opening else 0)
        if len(kills) >= 2 or clutch:
            scored.append((sc, r["round"], len(kills), clutch))
    chosen = sorted(sorted(scored, reverse=True)[:max_rounds], key=lambda x: x[1])
    if not chosen:                                                            # 实在没有多杀：取他击杀最多的回合
        best = max(st["rounds"], key=lambda r: sum(e["type"] == "kill" and e["attacker"] == fs for e in r["events"]))
        chosen = [(0, best["round"], 1, False)]
    keys = [c[1] for c in chosen]
    log(f"  高光回合 {keys}")
    tracks = {}
    for i, r in enumerate(r for r in st["rounds"] if r["round"] in keys):
        prog(0.55 + 0.15 * i / len(keys), f"生成高光回合的道具和走位 {i + 1}/{len(keys)}")
        if A.U is not None:
            r["utility"] = A.U.generate(st, r, team_ids, seed=seed + r["round"])
        cond, death, n = plan_tensors(A, st, r, team_ids)
        tracks[r["round"]] = A.M.generate(cond, map_name, n, death, seed=seed + r["round"], events=r["events"], phys=A.P, utility=r.get("utility"))
    meta = team_meta(teamA, teamB, team_ids, map_name, n_sims)
    meta["title"] = f"选手高光 · {MAP_CN[map_name]} · " + meta["title"].rsplit(" · ", 1)[-1]
    multi = sum(1 for r in st["rounds"] if sum(e["type"] == "kill" and e["attacker"] == fs for e in r["events"]) >= 3)
    fa, fb = st["final"]
    won = (fa > fb) == (q["team"] == "A")
    w, emit, n_frames = open_writer(out)
    with w:
        R = Renderer(A, st, res, meta, tracks, emit)
        cover = R.highlight_card(q, [f"本场 {q['kills']} 杀 {q['deaths']} 死", f"{multi} 次三杀以上 · {q['clutch_wins']} 次残局获胜",
                                     f"比赛 {max(fa, fb)}:{min(fa, fb)} {'获胜' if won else '惜败'}"], INTRO_FRAMES + 30)
        live = res["live_win_A"]
        for i, (_, rn, nk, clutch) in enumerate(chosen):
            r = next(x for x in st["rounds"] if x["round"] == rn)
            prog(0.72 + 0.25 * i / len(chosen), f"渲染高光 {i + 1}/{len(chosen)}：第 {rn} 回合")
            a, b = r["score_before"]
            after = (a + 1, b) if r["winner"] == "A" else (a, b + 1)
            R.key_round(r, tracks[rn], after, live, focus=fs)
        prog(0.98, "渲染结尾")
        R.highlight_card(q, [f"{q['kills']}-{q['deaths']} · 首杀 {q['opening_kills']} · 残局 {q['clutch_wins']}",
                             "★ 全场 MVP ★" if st["mvp"] == fs else f"{len(chosen)} 个高光回合"], OUTRO_FRAMES, big="高光集锦")
        plt.close(R.fig)
    imageio.imwrite(out.rsplit(".", 1)[0] + ".png", cover)
    json.dump({**{k: v for k, v in res.items() if k != "story"}, "story": st, "meta": meta, "focus": fs, "highlight_rounds": keys},
              open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1, default=float)
    log(f"  完成 {out}：{n_frames[0] / FPS:.0f} 秒视频，总用时 {time.time() - t0:.0f}s")
    return out, q, keys


# ======================================================================== Ban/Pick 与系列赛
class VetoRenderer:
    """Ban/Pick 动画（约 10 秒）和系列赛结算画面，和比赛画面同样的 9:16 风格。"""
    STEP, FLASH, BOARD, FINAL = 30, 8, 45, 90

    def __init__(self, meta, veto, map_wp, emit):
        self.meta, self.v, self.wp, self.emit = meta, veto, map_wp, emit
        self.fig = plt.figure(figsize=(10.8, 19.2), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        f = self.fig
        f.text(0.5, 0.982, "AI 模拟对局 · 非真实比赛", color="#ff6b6b", fontsize=20, ha="center", va="top", alpha=0.9)
        f.text(0.5, 0.955, meta["title"], color=DIM, fontsize=22, ha="center", va="top")
        fs = lambda name: 34 if len(name) <= 13 else max(20, int(34 * 13 / len(name)))
        f.text(0.04, 0.905, meta["a_name"], color=COL["A"], fontsize=fs(meta["a_name"]), fontweight="bold", ha="left", va="center")
        f.text(0.96, 0.905, meta["b_name"], color=COL["B"], fontsize=fs(meta["b_name"]), fontweight="bold", ha="right", va="center")
        f.text(0.5, 0.905, "VS", color=TXT, fontsize=40, fontweight="bold", ha="center", va="center")

    def frame(self):
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()

    def _clear(self, items):
        for t in items:
            t.remove()

    def veto(self):
        """逐步播放 ban/pick，返回最终总览帧（作封面）。"""
        f, v = self.fig, self.v
        fmt = v["fmt"].upper()
        items = [f.text(0.5, 0.855, f"{fmt} Ban / Pick", color=TXT, fontsize=44, fontweight="bold", ha="center", va="center"),
                 f.text(0.5, 0.825, "AI 按两队选手的选图习惯模拟（每步可能不同）", color=DIM, fontsize=19, ha="center", va="center")]
        pool = v["pool"]
        y_of = {m: 0.765 - i * 0.068 for i, m in enumerate(pool)}
        rows = {}
        for m in pool:
            y = y_of[m]
            box = FancyBboxPatch((0.08, y - 0.026), 0.84, 0.052, boxstyle="round,pad=0.004,rounding_size=0.012", transform=f.transFigure,
                                 fc=PANEL, ec="#2b3240", lw=2, zorder=1)
            f.add_artist(box)
            name = f.text(0.13, y, MAP_CN[m], color=TXT, fontsize=34, fontweight="bold", ha="left", va="center", zorder=2)
            tag = f.text(0.88, y, "", color=TXT, fontsize=26, fontweight="bold", ha="right", va="center", zorder=2)
            sub = f.text(0.50, y, "", color=DIM, fontsize=19, ha="center", va="center", zorder=2)
            rows[m] = (box, name, tag, sub)
        cap = f.text(0.5, 0.235, "", color=TXT, fontsize=34, fontweight="bold", ha="center", va="center")
        cap2 = f.text(0.5, 0.2, "", color=DIM, fontsize=21, ha="center", va="center")
        items += [cap, cap2]
        self.emit(self.frame(), self.BOARD)
        n_pick = 0
        for st in v["flow"]:
            m = st["map"]
            box, name, tag, sub = rows[m]
            who = st["team"]
            tname = self.meta["a_name" if who == "A" else "b_name"] if who else ""
            if st["action"] == "ban":
                box.set_alpha(0.45)
                name.set_color("#5b6270")
                tag.set_text(f"{'A' if who == 'A' else 'B'} 禁")
                tag.set_color(COL[who])
                strike = f.add_artist(plt.Line2D([0.12, 0.74], [y_of[m], y_of[m]], transform=f.transFigure, color="#ff4d4f", lw=3, alpha=0.8, zorder=3))
                items.append(strike)
                cap.set_text(f"{tname} 禁用 {MAP_CN[m]}")
                cap.set_color(COL[who])
            else:
                n_pick += 1
                label = f"第 {n_pick} 图" if st["action"] == "pick" else ("比赛地图" if v["fmt"] == "bo1" else "决胜图")
                if st["action"] == "pick":
                    box.set_edgecolor(COL[who])
                    tag.set_text(f"{label} · {'A' if who == 'A' else 'B'} 选")
                    tag.set_color(COL[who])
                    cap.set_text(f"{tname} 选择 {MAP_CN[m]}")
                    cap.set_color(COL[who])
                else:
                    box.set_edgecolor("#ffd666")
                    tag.set_text(label)
                    tag.set_color("#ffd666")
                    cap.set_text(f"剩下的 {MAP_CN[m]} 成为{label}")
                    cap.set_color("#ffd666")
                box.set_linewidth(4)
                if m in self.wp:
                    sub.set_text(f"{self.meta['a_short']} 胜率 {self.wp[m]:.0%}")
            cap2.set_text(f"AI 判断这一步的概率 {st['prob']:.0%}" if st["action"] != "decider" else "")
            box.set_facecolor("#24324a")                            # 闪一下
            self.emit(self.frame(), self.FLASH)
            box.set_facecolor(PANEL)
            self.emit(self.frame(), self.STEP - self.FLASH + (10 if st["action"] == "decider" else 0))
        order = v["order"]
        cap.set_text(f"{self.meta['a_short']} 系列赛胜率 {v.get('series_wp_flow', 0.5):.0%}" if v["fmt"] != "bo1"
                     else f"{MAP_CN[order[0]]} · {self.meta['a_short']} 胜率 {self.wp.get(order[0], 0.5):.0%}")
        cap.set_color(TXT)
        cap2.set_text("打图顺序：" + " → ".join(MAP_CN[m] for m in order))
        img = self.frame()
        self.emit(img, self.FINAL)
        self._clear(items)
        for m in pool:
            box, name, tag, sub = rows[m]
            box.remove()
            name.remove()
            tag.remove()
            sub.remove()
        return img

    def series_end(self, results, players):
        """results: [(图, 比分A, 比分B, 胜方)]；players: 系列赛累计数据 [{name, team, kills, deaths}]。"""
        f = self.fig
        sa = sum(r[3] == "A" for r in results)
        sb = len(results) - sa
        win = "A" if sa > sb else "B"
        items = [f.text(0.5, 0.80, f"{self.meta['a_name' if win == 'A' else 'b_name']}", color=COL[win], fontsize=46, fontweight="bold", ha="center"),
                 f.text(0.5, 0.75, f"{max(sa, sb)} : {min(sa, sb)} 赢下 {self.v['fmt'].upper()}", color=TXT, fontsize=40, fontweight="bold", ha="center")]
        for i, (m, a, b, w) in enumerate(results):
            items.append(f.text(0.5, 0.67 - i * 0.05, f"第 {i + 1} 图  {MAP_CN[m]}   {a} : {b}", color=COL[w], fontsize=30, ha="center"))
        mvp = max([q for q in players if q["team"] == win], key=lambda q: (q["kills"] - q["deaths"], q["kills"]))      # MVP 从获胜方选
        items.append(f.text(0.5, 0.67 - len(results) * 0.05 - 0.04, f"系列赛 MVP  {mvp['name']}  {mvp['kills']}-{mvp['deaths']}",
                            color="#ffd666", fontsize=32, fontweight="bold", ha="center"))
        items.append(f.text(0.5, 0.25, f"赛前 AI 预测 {self.meta['a_short']} 系列赛胜率 {self.v.get('series_wp', 0.5):.0%}", color=DIM, fontsize=22, ha="center"))
        self.emit(self.frame(), OUTRO_FRAMES)
        self._clear(items)

    def close(self):
        plt.close(self.fig)


def make_series_video(A, teamA, teamB, fmt, out, team_ids=(None, None), n_sims=400, seed=0, story="typical", preset=None, first=None,
                      key_k=3, log=print, progress=None):
    """先模拟 Ban/Pick（BO1/BO3/BO5），再按选出的地图逐图打完，合成一条视频。
    每张图谁赢按 P4 胜率先抽好（剧本编号决定），再挑对应的比赛剧本；dramatic 时尽量打满全部地图。"""
    import imageio.v2 as imageio
    if A.V is None:
        raise RuntimeError("还没有 Ban/Pick 模型，请先运行 p12_veto.py train")
    prog = progress or (lambda f, d: None)
    t0 = time.time()
    pool = A.V.preset(preset)["pool"]
    wp = {}
    for i, m in enumerate(pool):
        prog(0.03 * i / len(pool), f"AI 估算每张图的胜率：{MAP_CN[m]}")
        wp[m] = float(A.S4.simulate(teamA, teamB, m, n=1000, seed=seed)["p_win_A"])
    veto = A.V.simulate(teamA, teamB, fmt, preset, first, seed, map_wp=wp)
    need = {"bo1": 1, "bo3": 2, "bo5": 3}[fmt]
    rng = np.random.default_rng(seed + 7)
    for _ in range(200 if story == "dramatic" else 1):
        winners = []
        for m in veto["order"]:
            winners.append("A" if rng.random() < wp[m] else "B")
            if max(winners.count("A"), winners.count("B")) == need:
                break
        if story != "dramatic" or len(winners) == len(veto["order"]):
            break
    log(f"  Ban/Pick：{' → '.join(veto['order'])}；各图胜方 {winners}")
    n_maps = len(winners)
    meta0 = team_meta(teamA, teamB, team_ids, None, n_sims)
    meta0["title"] = f"梦幻对决 · {fmt.upper()} · " + meta0["title"].split(" · ", 1)[1]
    w, emit, n_frames = open_writer(out)
    stories, results, tot = [], [], {}
    with w:
        VR = VetoRenderer(meta0, veto, wp, emit)
        prog(0.03, "渲染 Ban/Pick")
        cover = VR.veto()
        sa = sb = 0
        span = 0.94 / n_maps
        for k, (m, win) in enumerate(zip(veto["order"], winners)):
            p0 = 0.05 + k * span
            res, st, keys, tracks = prepare_map(A, teamA, teamB, m, team_ids, n_sims, seed + 100 * (k + 1), story, key_k, winner=win,
                                                prog=sub_prog(lambda f, d, k=k: prog(f, f"第 {k + 1}/{n_maps} 图 {MAP_CN[m]}：{d}"), p0, p0 + 0.72 * span), log=log)
            fa, fb = st["final"]
            sa2, sb2 = (sa + 1, sb) if fa > fb else (sa, sb + 1)
            meta = team_meta(teamA, teamB, team_ids, m, n_sims)
            parts = []                                                       # 导播式系列赛条：已打完的图带比分，当前图标「进行中」，后面的图标谁选的
            picks = [(f_["map"], f_["team"]) for f_ in veto["flow"] if f_["action"] in ("pick", "decider")]
            for j, (mm, who) in enumerate(picks[:max(n_maps, k + 1)]):
                tag = (meta0["a_short"] if who == "A" else meta0["b_short"]) + " 选" if who else "决胜图"
                if j < k:
                    parts.append(f"{MAP_CN[mm]} {results[j][1]}-{results[j][2]}")
                elif j == k:
                    parts.append(f"{MAP_CN[mm]} 进行中（{tag}）")
                else:
                    parts.append(f"{MAP_CN[mm]}（{tag}）")
            meta["title"] = "  |  ".join(parts) if fmt != "bo1" else f"BO1 · {MAP_CN[m]}"
            meta["series_bar"] = fmt != "bo1"
            meta["series_after"] = f"系列赛 {sa2} : {sb2}" if fmt != "bo1" else ""
            render_map(A, res, st, tracks, meta, emit, sub_prog(lambda f, d, k=k: prog(f, f"第 {k + 1}/{n_maps} 图 {MAP_CN[m]}：{d}"), p0 + 0.72 * span, p0 + span))
            results.append((m, fa, fb, "A" if fa > fb else "B"))
            stories.append({**{x: y for x, y in res.items() if x != "story"}, "story": st, "key_rounds": keys})
            for q in st["players"]:
                t = tot.setdefault(q["slot"], {"name": q["name"], "team": q["team"], "kills": 0, "deaths": 0})
                t["kills"] += q["kills"]
                t["deaths"] += q["deaths"]
            sa, sb = sa2, sb2
        if fmt != "bo1":
            prog(0.99, "渲染系列赛结算")
            VR.series_end(results, list(tot.values()))
        VR.close()
    imageio.imwrite(out.rsplit(".", 1)[0] + ".png", cover)
    json.dump({"veto": veto, "map_wp": wp, "winners": winners, "results": results, "maps": stories, "meta": meta0},
              open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1, default=float)
    log(f"  完成 {out}：{n_frames[0] / FPS:.0f} 秒视频，{n_maps} 张图，总用时 {time.time() - t0:.0f}s")
    return out, veto, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--a-players")
    ap.add_argument("--b-players")
    ap.add_argument("--map", default="de_mirage")
    ap.add_argument("--out", default="out/video.mp4")
    ap.add_argument("--batch")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--story", default="typical", choices=["typical", "dramatic"])
    ap.add_argument("--series", choices=["bo1", "bo3", "bo5"], help="生成含 Ban/Pick 的系列赛（忽略 --map）")
    ap.add_argument("--pool", type=int, default=0, help="地图池预设编号，0 = 最新")
    ap.add_argument("--first", choices=["A", "B"], help="谁先 ban（默认随机）")
    ap.add_argument("--highlight", help="选手个人高光：选手名:年份（必须在两队阵容里），配合 --map")
    args = ap.parse_args()
    A = Assets()

    def side(team_spec, players_spec):
        if players_spec:
            return A.parse_players(players_spec), None
        slug, year = team_spec.rsplit(":", 1)
        return A.roster(slug, int(year)), (slug, int(year))
    if args.batch:
        df = pd.read_csv(args.batch)
        for i, r in df.iterrows():
            ta, ia = side(r["a"], None)
            tb, ib = side(r["b"], None)
            name = f"{r['a'].replace(':', '')}_vs_{r['b'].replace(':', '')}_{r['map'].replace('de_', '')}".replace("/", "-")
            print(f"[{i + 1}/{len(df)}] {name}", flush=True)
            sd = int(r.get("seed", args.seed)) if "seed" in df.columns else args.seed
            sty = r.get("story", args.story) if "story" in df.columns else args.story
            if str(r["map"]).lower() in ("bo1", "bo3", "bo5"):
                make_series_video(A, ta, tb, str(r["map"]).lower(), os.path.join(args.outdir, name + ".mp4"), (ia, ib), args.sims, sd, sty,
                                  preset=A.V.presets[args.pool]["label"], first=args.first)
            else:
                make_video(A, ta, tb, r["map"], os.path.join(args.outdir, name + ".mp4"), (ia, ib), args.sims, sd, sty)
    else:
        ta, ia = side(args.a, args.a_players)
        tb, ib = side(args.b, args.b_players)
        if args.highlight:
            n_, y_ = args.highlight.rsplit(":", 1)
            hit = A.by_name.get((n_.strip().lower(), int(y_)))
            if hit is None:
                raise SystemExit(f"找不到选手 {args.highlight}")
            make_highlight_video(A, ta, tb, args.map, args.out, (hit[0], int(y_)), (ia, ib), args.sims, args.seed)
        elif args.series:
            make_series_video(A, ta, tb, args.series, args.out, (ia, ib), args.sims, args.seed, args.story,
                              preset=A.V.presets[args.pool]["label"], first=args.first)
        else:
            make_video(A, ta, tb, args.map, args.out, (ia, ib), args.sims, args.seed, args.story)


if __name__ == "__main__":
    main()
