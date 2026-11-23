"""V7 地图物理：从真实比赛数据学每张图「哪里能走」「哪里能看到哪里」，并提供真实对枪的位置对。

- 可走网格（16 单位一格）：真实回合里有选手站过的格子（来自 11 万回合的逐秒走位）。
- 视线网格：可走格 ∪ 真实子弹路线经过的格子（攻击者 → 受害者连线，去掉穿墙击杀和道具/炸弹击杀）。
  墙、柱子、箱子内部既没人站过、也没有子弹穿过，所以是「不透明」的。
- 视线判断：两点连线上每隔 8 单位取点，全部落在视线网格里才算看得见（两端各留 16 单位容差）。
- 对枪位置对：真实击杀的 (攻击者位置, 受害者位置)，按 (攻击者点位, 受害者点位, 武器类别) 分桶，
  生成剧本时成对抽取，距离和角度都来自真实对枪。

- 视野：职业选手多用 4:3 拉伸分辨率，水平视野约 90°（朝向两侧各 45°）。敌人在视线内且落在视野里 = 「看见」。
- 接触：敌我双方有视线，且至少一方视野里有对方、或两人相距 R_AWARE 以内（近距离听得到脚步、余光看得到）。职业比赛里看见就会开枪，所以没有交火安排的接触是不真实的。
  统计「看见却没开枪」：接触发生后 CONTACT_WIN 秒内两人之间没有击杀、且 1 秒内两人都没参与任何击杀。

只看平面坐标（数据里没有高度），Nuke 等多层地图上下层重叠处会偏宽松。

用法：python p13_physics.py build      （写到 $CS2_MODELS/physics，并输出验证报告）
      python p13_physics.py contact    （统计真实比赛里「看见却没开枪」的频率，作为生成走位的对照）
"""
import paths
import json
import os
import sys
import time

import numpy as np
import pandas as pd

C = paths.CLEAN
MDIR = os.path.join(paths.MODELS, "physics")
CELL = 16.0
STEP = 8.0
END_TOL = 16.0
PAD = 200.0
FOV_HALF = 45.0
CONTACT_WIN = 2.5
SMOKE_R = 144.0        # CS2 烟雾半径约 144 单位
FIRE_R = 120.0         # 燃烧区域半径（近似）
R_AWARE = 500.0        # 近距离（约 9.5 米）有视线时，不管朝向都会发现对方（脚步声、余光），必然交火
NEAR = 1200.0          # 近距离接触（1200 单位以内，约 5 秒跑动距离）——观众最容易看出「看见却没开枪」

WCLASS = {}
for w in ("awp", "ssg08", "g3sg1", "scar20"):
    WCLASS[w] = "sniper"
for w in ("ak47", "m4a1_silencer", "m4a1", "galilar", "famas", "aug", "sg556", "m249", "negev"):
    WCLASS[w] = "rifle"
for w in ("mp9", "mac10", "mp7", "ump45", "p90", "mp5sd", "bizon"):
    WCLASS[w] = "smg"
for w in ("usp_silencer", "usp_silencer_off", "glock", "hkp2000", "p250", "tec9", "fiveseven", "deagle", "elite", "cz75a", "revolver"):
    WCLASS[w] = "pistol"
for w in ("xm1014", "mag7", "nova", "sawedoff"):
    WCLASS[w] = "shotgun"
for w in ("hegrenade", "inferno", "molotov", "incgrenade", "planted_c4", "world", "worldent", "smokegrenade", "flashbang", "decoy"):
    WCLASS[w] = "nade"           # 道具 / 炸弹 / 摔落：不需要视线


def weapon_class(w):
    w = str(w)
    if w.startswith("knife") or w == "bayonet":
        return "knife"
    if w == "taser":
        return "taser"
    return WCLASS.get(w, "rifle")


def needs_los(w):
    return weapon_class(w) != "nade"


# ---------------------------------------------------------------- 运行时
class Physics:
    def __init__(self, grids, meta, pairs):
        self.g, self.meta, self.pairs = grids, meta, pairs
        self.places = meta["places"]
        self.pid = {p: i for i, p in enumerate(self.places)}
        self._bucket = {}
        self._graph = {}
        self._field = {}
        self._trees = {}
        self.duel = meta.get("duel_stats", {})               # 从真实数据统计的对枪参数（受害者朝向攻击者的比例、接触到开枪的时间等）

    @classmethod
    def load(cls, mdir=None):
        mdir = mdir or MDIR
        meta = json.load(open(f"{mdir}/meta.json"))
        grids = dict(np.load(f"{mdir}/grids.npz"))
        pairs = dict(np.load(f"{mdir}/kill_pairs.npz"))
        return cls(grids, meta, pairs)

    def has(self, m):
        return f"{m}_walk" in self.g

    def _cells(self, m, x, y):
        o = self.meta["origin"][m]
        g = self.g[f"{m}_walk"]
        i = np.clip(((np.asarray(x) - o[0]) / CELL).astype(int), 0, g.shape[0] - 1)
        j = np.clip(((np.asarray(y) - o[1]) / CELL).astype(int), 0, g.shape[1] - 1)
        return i, j

    def walkable(self, m, x, y):
        if not self.has(m):
            return np.ones(np.shape(x), bool)
        i, j = self._cells(m, x, y)
        return self.g[f"{m}_walk"][i, j]

    def path_walkable(self, m, x0, y0, x1, y1, n=6):
        """一秒内的移动路径（起点→终点）上取 n 个点，全部可走才行——防止一步跨过薄墙。"""
        if not self.has(m):
            return np.ones(np.shape(x1), bool)
        t = np.linspace(0, 1, n + 1)[1:]
        xs = np.asarray(x0)[..., None] + (np.asarray(x1) - np.asarray(x0))[..., None] * t
        ys = np.asarray(y0)[..., None] + (np.asarray(y1) - np.asarray(y0))[..., None] * t
        return self.walkable(m, xs, ys).all(-1)

    def los(self, m, ax, ay, vx, vy):
        """两点之间是否有视线（可批量）。"""
        if not self.has(m):
            return np.ones(np.shape(ax), bool)
        ax, ay, vx, vy = (np.atleast_1d(np.asarray(z, float)) for z in (ax, ay, vx, vy))
        d = np.hypot(vx - ax, vy - ay)
        out = np.ones(len(ax), bool)
        see = self.g[f"{m}_see"]
        for k in np.where(d > 2 * END_TOL)[0]:
            n = int((d[k] - 2 * END_TOL) / STEP) + 1
            t = (END_TOL + np.arange(n) * STEP) / d[k]
            i, j = self._cells(m, ax[k] + (vx[k] - ax[k]) * t, ay[k] + (vy[k] - ay[k]) * t)
            out[k] = see[i, j].all()
        return out

    def los_many(self, m, ax, ay, vx, vy):
        """一个攻击者位置 vs 多个候选位置（或反之）的视线，向量化版本。"""
        if not self.has(m):
            return np.ones(np.broadcast(ax, vx).shape, bool)
        ax, ay, vx, vy = np.broadcast_arrays(*(np.asarray(z, float) for z in (ax, ay, vx, vy)))
        d = np.hypot(vx - ax, vy - ay)
        n = int(max(1, (d.max() - 2 * END_TOL) / STEP + 1))
        s = END_TOL + np.arange(n) * STEP                                   # 沿线距离
        dd = np.maximum(d, 1e-6)[..., None]
        t = s / dd
        valid = s < (d[..., None] - END_TOL)
        i, j = self._cells(m, ax[..., None] + (vx - ax)[..., None] * t, ay[..., None] + (vy - ay)[..., None] * t)
        ok = self.g[f"{m}_see"][i, j] | ~valid
        return ok.all(-1)

    # ---- 真实枪线检索：所有候选枪线都来自真实击杀，不凭空构造
    def _tree(self, m, kind):
        key = (m, kind)
        if key not in self._trees:
            from scipy.spatial import cKDTree
            P = self.pairs
            pts = {"av": np.hstack([P[f"{m}_a"], P[f"{m}_v"]]), "a": P[f"{m}_a"], "v": P[f"{m}_v"]}[kind]
            self._trees[key] = cKDTree(pts)
        return self._trees[key]

    def real_lines_near(self, m, a_xy=None, v_xy=None, k=300, weapon=None):
        """离 (攻击者位置, 受害者位置) 最近的 k 条真实枪线；只给一端时按那一端检索。返回 (攻击者位置[n,2], 受害者位置[n,2])。"""
        if f"{m}_a" not in self.pairs:
            return np.zeros((0, 2)), np.zeros((0, 2))
        P = self.pairs
        if a_xy is not None and v_xy is not None:
            _, idx = self._tree(m, "av").query(np.r_[a_xy, v_xy], k=min(k * 3, len(P[f"{m}_a"])))
        elif v_xy is not None:
            _, idx = self._tree(m, "v").query(np.asarray(v_xy), k=min(k * 3, len(P[f"{m}_a"])))
        else:
            _, idx = self._tree(m, "a").query(np.asarray(a_xy), k=min(k * 3, len(P[f"{m}_a"])))
        idx = np.atleast_1d(idx)
        if weapon is not None:
            wi = self.meta["wclasses"].index(weapon_class(weapon)) if weapon_class(weapon) in self.meta["wclasses"] else -1
            same = idx[P[f"{m}_wc"][idx] == wi]
            if len(same) >= 20:
                idx = same
        idx = idx[:k]
        return P[f"{m}_a"][idx].astype(float), P[f"{m}_v"][idx].astype(float)

    # ---- 绕墙路程（可走网格上的最短路）
    def _walk_graph(self, m):
        if m not in self._graph:
            from scipy.sparse import coo_matrix
            g = self.g[f"{m}_walk"]
            nx, ny = g.shape
            idx = -np.ones(g.shape, np.int64)
            cells = np.argwhere(g)
            idx[g] = np.arange(len(cells))
            rows, cols, wts = [], [], []
            for di, dj, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142)):
                a = idx[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
                b = idx[max(0, di):nx + min(0, di), max(0, dj):ny + min(0, dj)]
                ok = (a >= 0) & (b >= 0)
                rows.append(a[ok]); cols.append(b[ok]); wts.append(np.full(ok.sum(), w * CELL))
            r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
            n = len(cells)
            self._graph[m] = (coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr(), idx)
        return self._graph[m]

    def geo_field(self, m, x, y):
        """从 (x,y) 出发、沿可走区域到每个格子的最短路程（单位：游戏单位）；不可走格为 inf。按格缓存。"""
        if not self.has(m):
            return None
        i, j = self._cells(m, x, y)
        key = (m, int(i), int(j))
        if key not in self._field:
            from scipy.sparse.csgraph import dijkstra
            G, idx = self._walk_graph(m)
            src = idx[i, j]
            if src < 0:                                                      # 目标点不在可走格上：找最近的可走格
                cells = np.argwhere(idx >= 0)
                src = idx[tuple(cells[np.argmin((cells[:, 0] - i) ** 2 + (cells[:, 1] - j) ** 2)])]
            d = dijkstra(G, indices=int(src), limit=6000.0)
            f = np.full(idx.shape, np.inf, np.float32)
            f[idx >= 0] = d[idx[idx >= 0]]
            if len(self._field) > 400:
                self._field.clear()
            self._field[key] = f
        return self._field[key]

    def geo_lookup(self, m, field, x, y):
        i, j = self._cells(m, x, y)
        return field[i, j]

    def sample_pair(self, m, a_place, v_place, weapon, rng, near_a=None, near_v=None):
        """按 (攻击者点位, 受害者点位, 武器类别) 抽一组真实对枪位置；桶空时逐级放宽。返回 ([ax,ay], [vx,vy]) 或 None。
        near_a / near_v：(x, y, 允许偏离距离)——同一个人短时间内连续交火时，下一组位置要离他上一个位置近（不能瞬移）。"""
        if f"{m}_a" not in self.pairs:
            return None
        wc = weapon_class(weapon)
        ap, vp = self.pid.get(a_place, -1), self.pid.get(v_place, -1)
        P = self.pairs
        keys = [("apvw", ap, vp, wc), ("apv", ap, vp, None), ("vw", -1, vp, wc), ("v", -1, vp, None), ("aw", ap, -1, wc)]
        for name, a_, v_, w_ in keys:
            k = (m, name, a_, v_, w_)
            if k not in self._bucket:
                msk = np.ones(len(P[f"{m}_a"]), bool)
                if a_ != -1:
                    msk &= P[f"{m}_ap"] == a_
                if v_ != -1:
                    msk &= P[f"{m}_vp"] == v_
                if w_ is not None:
                    msk &= P[f"{m}_wc"] == self.meta["wclasses"].index(w_)
                self._bucket[k] = np.where(msk)[0]
            idx = self._bucket[k]
            if (name in ("apvw", "apv") and len(idx) >= 1) or len(idx) >= 5:
                if near_a is None and near_v is None:
                    r = idx[rng.integers(len(idx))]
                else:
                    cand = idx[rng.choice(len(idx), min(len(idx), 400), replace=False)]
                    pen = np.zeros(len(cand))
                    for near, key in ((near_a, "_a"), (near_v, "_v")):
                        if near is not None:
                            dd = np.hypot(P[f"{m}{key}"][cand, 0] - near[0], P[f"{m}{key}"][cand, 1] - near[1])
                            pen += np.maximum(0, dd - near[2]) / 100.0
                    if name in ("apvw", "apv") and pen.min() > 5 and len(keys) > 2:
                        continue                                              # 这个桶里没有离得近的，放宽点位条件再找
                    r = cand[np.argmin(pen + rng.uniform(0, 0.5, len(cand)))]
                a, v = P[f"{m}_a"][r], P[f"{m}_v"][r]
                return [float(a[0]), float(a[1])], [float(v[0]), float(v[1])]
        return None


def faces(yaw_deg, fx, fy, tx, ty, half=FOV_HALF):
    """站在 (fx,fy)、朝向 yaw 的人，视野里是否有 (tx,ty)。"""
    ang = np.degrees(np.arctan2(np.asarray(ty) - fy, np.asarray(tx) - fx))
    diff = (ang - np.asarray(yaw_deg) + 180.0) % 360.0 - 180.0
    return np.abs(diff) <= half


def pair_allowed(kills, step, i, j, win=CONTACT_WIN):
    """第 step 秒 i、j 互相看见是否「有交火解释」：win 秒内两人之间有击杀，或 1 秒内两人之一参与了击杀（交叉火力/补枪）。
    kills: [(时间秒, 攻击者, 受害者)]"""
    for t, a, v in kills:
        if {a, v} == {i, j} and step - 1.0 <= t <= step + win:
            return True
        if (a in (i, j) or v in (i, j)) and abs(t - step) <= 1.0:
            return True
    return False


def seg_hits_circles(ax, ay, bx, by, circles, r):
    """线段 (ax,ay)-(bx,by)（可批量）是否穿过任一圆（circles: [(x, y)]）。"""
    ax, ay, bx, by = (np.asarray(z, float) for z in (ax, ay, bx, by))
    hit = np.zeros(np.broadcast(ax, bx).shape, bool)
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-6)
    for cx, cy in circles:
        t = np.clip(((cx - ax) * dx + (cy - ay) * dy) / L2, 0, 1)
        hit |= np.hypot(ax + t * dx - cx, ay + t * dy - cy) < r
    return hit


def contacts(phys, m, xy, yaw, alive, smokes=()):
    """某一时刻的接触对：返回 [(i, j)]（i 在 0–4 号，j 在 5–9 号），双方都活着、有视线、且至少一方视野里有对方或两人很近。"""
    out = []
    A = [i for i in range(5) if alive[i]]
    B = [j for j in range(5, 10) if alive[j]]
    if not A or not B:
        return out
    ii = np.repeat(A, len(B))
    jj = np.tile(B, len(A))
    see = phys.los_many(m, xy[ii, 0], xy[ii, 1], xy[jj, 0], xy[jj, 1])
    if len(smokes):                                                         # 隔着烟看不见
        see &= ~seg_hits_circles(xy[ii, 0], xy[ii, 1], xy[jj, 0], xy[jj, 1], smokes, SMOKE_R)
    f1 = faces(yaw[ii], xy[ii, 0], xy[ii, 1], xy[jj, 0], xy[jj, 1])
    f2 = faces(yaw[jj], xy[jj, 0], xy[jj, 1], xy[ii, 0], xy[ii, 1])
    near = np.hypot(xy[ii, 0] - xy[jj, 0], xy[ii, 1] - xy[jj, 1]) < R_AWARE
    for k in np.where(see & (f1 | f2 | near))[0]:
        out.append((int(ii[k]), int(jj[k])))
    return out


def contact_stats(phys, m, tr, kills, need_los=None, utility=None):
    """一回合走位 tr [n,10,4]（x,y,yaw,alive）+ 击杀 [(t,a,v)] → 统计。
    返回：接触对·秒数、其中没有交火解释的数、需要视线的击杀里开枪瞬间隔墙的数、活着的敌我对·秒数。"""
    n = len(tr)
    c_all = c_bad = c_near = pair_sec = 0
    sm_at = lambda t: [(u["x"], u["y"]) for u in (utility or []) if u["type"] == "smoke" and u["t_start"] <= t <= u["t_end"]]   # 烟雾挡视线（与生成时一致）
    for s in range(1, n):
        al = tr[s, :, 3] > 0
        pair_sec += int(al[:5].sum() * al[5:].sum())
        mid = 0.5 * (tr[s - 1, :, :2] + tr[s, :, :2])               # 视频里的半秒处：插值位置 + 前一秒的朝向
        found = {(i, j): tr[s, :, :2] for i, j in contacts(phys, m, tr[s, :, :2], tr[s, :, 2], al, sm_at(s))}
        for i, j in contacts(phys, m, mid, tr[s - 1, :, 2], al, sm_at(s - 0.5)):
            found.setdefault((i, j), mid)
        for (i, j), xy in found.items():
            c_all += 1
            if not pair_allowed(kills, s, i, j):
                c_bad += 1
                c_near += int(np.hypot(*(xy[i] - xy[j])) < NEAR)
    wall = 0
    nk = 0
    for idx, (t, a, v) in enumerate(kills):
        if need_los is not None and not need_los[idx]:
            continue
        if t < 0:
            continue
        k = min(int(t), n - 1)
        k2 = min(k + 1, n - 1)
        fr = t - int(t)
        # 死后的位置无效：两人都活着的那一秒才插值，否则用击杀前一秒（第 k 秒）的位置
        if tr[k2, a, 3] > 0 and tr[k2, v, 3] > 0:
            pa = tr[k, a, :2] * (1 - fr) + tr[k2, a, :2] * fr
            pv = tr[k, v, :2] * (1 - fr) + tr[k2, v, :2] * fr
        else:
            pa, pv = tr[k, a, :2], tr[k, v, :2]
        nk += 1
        wall += int(not phys.los(m, pa[0], pa[1], pv[0], pv[1])[0])
    return {"contact": c_all, "contact_no_fight": c_bad, "contact_no_fight_near": c_near, "pair_sec": pair_sec, "kills_los": nk, "kills_through_wall": wall}


def build_height_main():
    """地形高度：830 万条带高度的真实站位，按 16 单位格取高度（多层地图取站人最多的那一层）。写进 grids.npz 的 {图}_height。"""
    meta = json.load(open(f"{MDIR}/meta.json"))
    grids = dict(np.load(f"{MDIR}/grids.npz"))
    pos = pd.read_parquet(f"{C}/positions.parquet", columns=["map_name", "X", "Y", "Z", "place"])
    for m, g in pos.groupby("map_name"):
        if f"{m}_walk" not in grids:
            continue
        o = meta["origin"][m]
        nx, ny = grids[f"{m}_walk"].shape
        i = np.clip(((g.X.values - o[0]) / CELL).astype(int), 0, nx - 1)
        j = np.clip(((g.Y.values - o[1]) / CELL).astype(int), 0, ny - 1)
        cid = i * ny + j
        # 多层地图同一格有上下两层：按 48 单位分层，取站人最多的那一层，再在该层内取中位数
        zb = np.round(g.Z.values / 48.0).astype(np.int64)
        df = pd.DataFrame({"c": cid, "zb": zb, "z": g.Z.values})
        cnt = df.groupby(["c", "zb"]).size().rename("n").reset_index()
        top = cnt.sort_values("n").drop_duplicates("c", keep="last")
        df = df.merge(top[["c", "zb"]], on=["c", "zb"])
        med = df.groupby("c").z.median()
        h = np.full(nx * ny, np.nan, np.float32)
        h[med.index.values] = med.values
        grids[f"{m}_height"] = h.reshape(nx, ny)
        # 包点区域：站位数据里每个位置都有区域名，取每格出现最多的区域，是 BombsiteA / BombsiteB 的格子标 1 / 2
        pl = pd.DataFrame({"c": cid, "p": g.place.values})
        pl = pl[pl.p.isin(["BombsiteA", "BombsiteB"])]
        site = np.zeros(nx * ny, np.int8)
        if len(pl):
            tot = pd.Series(cid).value_counts()
            sc = pl.groupby(["c", "p"]).size().rename("n").reset_index()
            sc = sc[sc.n >= 0.5 * tot.reindex(sc.c).values]
            site[sc.c.values] = np.where(sc.p.values == "BombsiteA", 1, 2)
        grids[f"{m}_site"] = site.reshape(nx, ny)
        print(f"  {m}: 高度范围 {np.nanpercentile(h, 2):.0f} ~ {np.nanpercentile(h, 98):.0f}", flush=True)
    np.savez_compressed(f"{MDIR}/grids.npz", **grids)


def duel_stats_main(n_kills=5000):
    """从真实比赛统计对枪参数，写进 meta.json 的 duel_stats，生成走位时直接使用（不手工设定）：
    - 攻击者 / 受害者在击杀时朝向对方的比例（逐秒走位的朝向，视野 ±45°）；
    - 击杀前双方连续处于「接触」（有视线且至少一方看见对方）的秒数分布 → 「看见之后多久必然开枪」的时间窗口。"""
    phys = Physics.load()
    ti = pd.read_parquet(f"{C}/tracks_index.parquet")
    ti["map_name"] = ti.map_uid.str.split("_", n=1).str[1]
    T = np.load(f"{C}/tracks.npy", mmap_mode="r")
    ev = pd.read_parquet(f"{C}/events.parquet", columns=["type", "weapon", "penetrated", "a_slot", "v_slot", "map_uid", "real_round", "time"])
    k = ev[(ev.type == "kill") & (ev.time >= 1)].copy()
    k["penetrated"] = k.penetrated.fillna(False).astype(bool)
    k = k[~k.penetrated & k.weapon.map(needs_los) & k.a_slot.between(0, 9) & k.v_slot.between(0, 9)]
    k = k.merge(ti, on=["map_uid", "real_round"]).sample(n_kills, random_state=0)
    fa, fv, dur = [], [], []
    for r in k.itertuples():
        s0 = min(int(r.time), r.n_steps - 1)
        tr = np.asarray(T[r.row, max(0, s0 - 12):s0 + 1], dtype=np.float32)
        a, v = int(r.a_slot), int(r.v_slot)
        last = tr[-1]
        fa.append(bool(faces(last[a, 2], last[a, 0], last[a, 1], last[v, 0], last[v, 1])))
        fv.append(bool(faces(last[v, 2], last[v, 0], last[v, 1], last[a, 0], last[a, 1])))
        d = 0
        for row in tr[::-1]:
            if row[a, 3] <= 0 or row[v, 3] <= 0:
                break
            see = phys.los(r.map_name, row[a, 0], row[a, 1], row[v, 0], row[v, 1])[0]
            if not (see and (faces(row[a, 2], row[a, 0], row[a, 1], row[v, 0], row[v, 1]) or faces(row[v, 2], row[v, 0], row[v, 1], row[a, 0], row[a, 1]))):
                break
            d += 1
        dur.append(d)
    dur = np.array(dur)
    st = {"样本击杀数": int(len(k)), "攻击者朝向受害者": round(float(np.mean(fa)), 3), "受害者朝向攻击者": round(float(np.mean(fv)), 3),
          "击杀前连续接触秒数": {"中位数": float(np.median(dur)), "75%": float(np.percentile(dur, 75)), "90%": float(np.percentile(dur, 90))},
          # 取 75% 分位 + 1 秒：长尾主要来自平面近似（没有烟雾和高度信息时会把实际看不见的情况算成「看见」）
          "接触后开枪窗口（秒）": float(np.percentile(dur, 75)) + 1.0, "窗口取法": "击杀前连续接触秒数的 75% 分位 + 1 秒"}
    meta = json.load(open(f"{MDIR}/meta.json"))
    meta["duel_stats"] = st
    json.dump(meta, open(f"{MDIR}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False))
    return st


def real_contact_main(n_rounds=400):
    """真实比赛的对照：随机抽 n_rounds 个回合，统计接触与「看见却没开枪」。"""
    phys = Physics.load()
    ti = pd.read_parquet(f"{C}/tracks_index.parquet")
    ti["map_name"] = ti.map_uid.str.split("_", n=1).str[1]
    T = np.load(f"{C}/tracks.npy", mmap_mode="r")
    ev = pd.read_parquet(f"{C}/events.parquet", columns=["type", "weapon", "penetrated", "a_slot", "v_slot", "map_uid", "real_round", "time"])
    k = ev[(ev.type == "kill")].copy()
    k["penetrated"] = k.penetrated.fillna(False).astype(bool)
    rng = np.random.default_rng(0)
    pick = ti.iloc[rng.choice(len(ti), n_rounds, replace=False)]
    kg = {key: g for key, g in k[k.map_uid.isin(set(pick.map_uid))].groupby(["map_uid", "real_round"])}
    tot = {"contact": 0, "contact_no_fight": 0, "contact_no_fight_near": 0, "pair_sec": 0, "kills_los": 0, "kills_through_wall": 0}
    for r in pick.itertuples():
        tr = np.asarray(T[r.row, :r.n_steps], dtype=np.float32)
        g = kg.get((r.map_uid, r.real_round))
        kills, need = [], []
        if g is not None:
            for e in g.itertuples():
                if 0 <= e.a_slot < 10 and 0 <= e.v_slot < 10:
                    kills.append((float(e.time), int(e.a_slot), int(e.v_slot)))
                    need.append(needs_los(e.weapon) and not e.penetrated)
        st = contact_stats(phys, r.map_name, tr, kills, need)
        for kk in tot:
            tot[kk] += st[kk]
    res = summarize_contacts(tot, n_rounds)
    print(json.dumps(res, ensure_ascii=False))
    rep = json.load(open(f"{MDIR}/physics_report.json"))
    rep["真实比赛的接触统计"] = res
    json.dump(rep, open(f"{MDIR}/physics_report.json", "w"), ensure_ascii=False, indent=1)
    return res


def summarize_contacts(tot, n_rounds):
    return {"回合数": n_rounds, "每回合接触对·秒": round(tot["contact"] / n_rounds, 2),
            "每回合「看见却没开枪」对·秒": round(tot["contact_no_fight"] / n_rounds, 2),
            "其中近距离（<1200）": round(tot["contact_no_fight_near"] / n_rounds, 2),
            "「看见却没开枪」占活着的敌我对·秒": round(tot["contact_no_fight"] / max(1, tot["pair_sec"]), 4),
            "开枪瞬间隔墙的击杀比例": round(tot["kills_through_wall"] / max(1, tot["kills_los"]), 3), "需要视线的击杀数": tot["kills_los"]}


def eval_generation_main(maps=("de_mirage", "de_inferno", "de_nuke"), seed=0):
    """同一批剧本，对比「无物理约束」和「有物理约束」生成的走位：开枪瞬间隔墙、看见却没开枪、击杀距离。"""
    import render_video as rv
    A = rv.Assets()
    teams = [(A.roster("vitality", 2025), ("vitality", 2025)), (A.roster("spirit", 2026), ("spirit", 2026))]
    real_d = {}
    for m in maps:
        P = A.P.pairs
        if f"{m}_a" in P:
            d = np.hypot(*(P[f"{m}_a"] - P[f"{m}_v"]).T)
            for wi, wc in enumerate(A.P.meta["wclasses"]):
                real_d.setdefault(wc, []).extend(d[P[f"{m}_wc"] == wi].tolist())
    res = {"无物理约束": {}, "有物理约束": {}}
    dist = {"无物理约束": {}, "有物理约束": {}}
    real_hit = {"无物理约束": [], "有物理约束": []}
    cstats = {}
    n_rounds = 0
    t0 = time.time()
    for mi_, m in enumerate(maps):
        out = A.G.generate_match(teams[0][0], teams[1][0], m, n_sims=60, seed=seed + mi_, team_ids=(teams[0][1], teams[1][1]), guide=A.S4)
        st = out["story"]
        for r in st["rounds"]:
            cond, death, n = rv.plan_tensors(A, st, r, (teams[0][1], teams[1][1]))
            kills = [(e["time"], e["attacker"], e["victim"]) for e in r["events"] if e["type"] == "kill"]
            need = [needs_los(e.get("weapon", "")) for e in r["events"] if e["type"] == "kill"]
            wcs = [weapon_class(e.get("weapon", "")) for e in r["events"] if e["type"] == "kill"]
            util = A.U.generate(st, r, (teams[0][1], teams[1][1]), seed=seed + r["round"]) if getattr(A, "U", None) is not None else None
            for tag, kw in (("无物理约束", {}), ("有物理约束", {"events": r["events"], "phys": A.P, "stats": cstats, "utility": util})):
                tr = A.M.generate(cond, m, n, death, seed=seed + r["round"], **kw)
                stt = contact_stats(A.P, m, tr, kills, need, kw.get("utility"))
                for kk, vv in stt.items():
                    res[tag][kk] = res[tag].get(kk, 0) + vv
                for (t, a, v), wc, nd in zip(kills, wcs, need):
                    k = min(int(t), n - 1)
                    dist[tag].setdefault(wc, []).append(float(np.hypot(*(tr[k, a, :2] - tr[k, v, :2]))))
                    if nd and t >= 0:
                        dq, _ = A.P._tree(m, "av").query(np.r_[tr[k, a, :2], tr[k, v, :2]])
                        real_hit[tag].append(dq <= 150)
            n_rounds += 1
        print(f"  {m}: {len(st['rounds'])} 回合，{time.time() - t0:.0f}s", flush=True)
    report = {tag: summarize_contacts(v, n_rounds) for tag, v in res.items()}
    report["约束触发次数"] = {k_: v_ for k_, v_ in cstats.items() if k_ != "fail_detail"}
    report["枪线落在真实枪线附近的比例（两端 4 维距离 ≤ 150）"] = {tag: round(float(np.mean(v)), 3) for tag, v in real_hit.items()}
    report["击杀距离中位数（按武器）"] = {wc: {"真实": round(float(np.median(real_d[wc])), 0) if real_d.get(wc) else None,
                                      **{tag: round(float(np.median(dist[tag][wc])), 0) if dist[tag].get(wc) else None for tag in dist}}
                                 for wc in ("sniper", "rifle", "smg", "pistol")}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    rep = json.load(open(f"{MDIR}/physics_report.json"))
    rep["生成走位对比"] = report
    json.dump(rep, open(f"{MDIR}/physics_report.json", "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 构建
def build_main():
    t0 = time.time()
    os.makedirs(MDIR, exist_ok=True)
    bounds = json.load(open(f"{C}/map_bounds.json"))
    ti = pd.read_parquet(f"{C}/tracks_index.parquet")
    ti["map_name"] = ti.map_uid.str.split("_", n=1).str[1]
    T = np.load(f"{C}/tracks.npy", mmap_mode="r")
    ev = pd.read_parquet(f"{C}/events.parquet", columns=["type", "weapon", "penetrated", "ax", "ay", "vx", "vy", "a_place", "v_place", "map_uid"])
    k = ev[(ev.type == "kill")].dropna(subset=["ax", "ay", "vx", "vy"]).copy()
    k["map_name"] = k.map_uid.str.split("_", n=1).str[1]
    k["wc"] = k.weapon.map(weapon_class)
    k["penetrated"] = k.penetrated.fillna(False).astype(bool)          # 原始列是可空布尔（object），先转成普通布尔
    # 留出 15% 的地图验证视线判断（构建网格时不用它们的子弹路线和站位）
    uids = np.array(sorted(k.map_uid.unique()))
    rng = np.random.default_rng(0)
    hold = set(rng.choice(uids, int(len(uids) * 0.15), replace=False))
    places = sorted(set(k.a_place.dropna()) | set(k.v_place.dropna()))
    pid = {p: i for i, p in enumerate(places)}
    wcl = ["sniper", "rifle", "smg", "pistol", "shotgun", "knife", "taser", "nade"]
    grids, origin, pairs, report = {}, {}, {}, {}
    for m, b in bounds.items():
        x0, y0 = b[0] - PAD, b[2] - PAD
        nx, ny = int((b[1] - b[0] + 2 * PAD) / CELL) + 1, int((b[3] - b[2] + 2 * PAD) / CELL) + 1
        origin[m] = [x0, y0]

        def cell_idx(x, y):
            i = np.clip(((x - x0) / CELL).astype(np.int64), 0, nx - 1)
            j = np.clip(((y - y0) / CELL).astype(np.int64), 0, ny - 1)
            return i * ny + j

        res = {}
        for tag, excl in (("eval", hold), ("full", set())):
            # 可走：逐秒站位
            rows = ti[(ti.map_name == m) & ~ti.map_uid.isin(excl)].row.values
            cnt = np.zeros(nx * ny, np.int64)
            for s in range(0, len(rows), 2000):
                tr = np.asarray(T[np.sort(rows[s:s + 2000])], dtype=np.float32)
                p = tr[..., :2][tr[..., 3] > 0]
                cnt += np.bincount(cell_idx(p[:, 0], p[:, 1]), minlength=nx * ny)
            walk = (cnt >= 3).reshape(nx, ny)
            # 子弹路线：非穿墙、需要视线的击杀
            kk = k[(k.map_name == m) & ~k.map_uid.isin(excl) & ~k.penetrated & (k.wc != "nade")]
            lc = np.zeros(nx * ny, np.int64)
            A = kk[["ax", "ay"]].values
            V = kk[["vx", "vy"]].values
            d = np.hypot(*(V - A).T)
            for s in range(0, len(kk), 20000):
                a, v, dd = A[s:s + 20000], V[s:s + 20000], d[s:s + 20000]
                n = (dd / STEP).astype(int) + 1
                rep = np.repeat(np.arange(len(a)), n)
                off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
                t = np.minimum(off * STEP / np.maximum(dd[rep], 1e-6), 1.0)
                px = a[rep, 0] + (v[rep, 0] - a[rep, 0]) * t
                py = a[rep, 1] + (v[rep, 1] - a[rep, 1]) * t
                # 每条线在每个格子只记一次
                ci = np.unique(rep.astype(np.int64) * (nx * ny) + cell_idx(px, py)) % (nx * ny)
                lc += np.bincount(ci, minlength=nx * ny)
            see = walk | (lc >= 3).reshape(nx, ny)
            res[tag] = (walk, see)
        grids[f"{m}_walk"], grids[f"{m}_see"] = res["full"]
        # ---- 验证（用 eval 网格 + 留出地图）
        phys_eval = Physics({f"{m}_walk": res["eval"][0], f"{m}_see": res["eval"][1]}, {"origin": {m: origin[m]}, "places": places, "wclasses": wcl}, {})
        hk = k[(k.map_name == m) & k.map_uid.isin(hold) & (k.wc != "nade")]
        if len(hk) > 200:
            hk = hk.sample(min(len(hk), 4000), random_state=0)
            clean = hk[~hk.penetrated]
            pen = hk[hk.penetrated]
            ok_clean = phys_eval.los(m, clean.ax.values, clean.ay.values, clean.vx.values, clean.vy.values).mean()
            ok_pen = phys_eval.los(m, pen.ax.values, pen.ay.values, pen.vx.values, pen.vy.values).mean() if len(pen) else float("nan")
            # 随机配对：攻击者位置和另一次击杀的受害者位置（距离分布相近的真实站位，但不保证互相看得见）
            sh = clean.sample(frac=1.0, random_state=1)
            ok_rand = phys_eval.los(m, clean.ax.values, clean.ay.values, sh.vx.values, sh.vy.values).mean()
            report[m] = {"留出击杀数": int(len(clean)), "真实非穿墙击杀判为可见": round(float(ok_clean), 3),
                         "真实穿墙击杀判为可见": round(float(ok_pen), 3) if ok_pen == ok_pen else None, "穿墙击杀数": int(len(pen)),
                         "随机配对判为可见": round(float(ok_rand), 3), "可走格": int(res["full"][0].sum()), "视线格": int(res["full"][1].sum())}
        # ---- 对枪位置对（全部数据，非穿墙且需要视线的击杀，且在完整网格上确实看得见）
        full = Physics({f"{m}_walk": res["full"][0], f"{m}_see": res["full"][1]}, {"origin": {m: origin[m]}, "places": places, "wclasses": wcl}, {})
        kk = k[(k.map_name == m) & ~k.penetrated & (k.wc != "nade")]
        if len(kk):
            vis = full.los(m, kk.ax.values, kk.ay.values, kk.vx.values, kk.vy.values)
            kk = kk[vis]
            pairs[f"{m}_a"] = kk[["ax", "ay"]].values.astype(np.float32)
            pairs[f"{m}_v"] = kk[["vx", "vy"]].values.astype(np.float32)
            pairs[f"{m}_ap"] = kk.a_place.map(pid).fillna(-1).astype(np.int32).values
            pairs[f"{m}_vp"] = kk.v_place.map(pid).fillna(-1).astype(np.int32).values
            pairs[f"{m}_wc"] = kk.wc.map(wcl.index).astype(np.int8).values
        print(f"  {m}: {report.get(m)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{MDIR}/grids.npz", **grids)
    np.savez_compressed(f"{MDIR}/kill_pairs.npz", **pairs)
    json.dump({"origin": origin, "cell": CELL, "places": places, "wclasses": wcl, "report": report}, open(f"{MDIR}/meta.json", "w"), ensure_ascii=False, indent=1)
    tot = pd.DataFrame(report).T
    w = tot["留出击杀数"]
    summ = {"真实非穿墙击杀判为可见（加权）": round(float((tot["真实非穿墙击杀判为可见"] * w).sum() / w.sum()), 3),
            "真实穿墙击杀判为可见（加权）": round(float((tot["真实穿墙击杀判为可见"].astype(float) * tot["穿墙击杀数"]).sum() / tot["穿墙击杀数"].sum()), 3),
            "随机配对判为可见（加权）": round(float((tot["随机配对判为可见"] * w).sum() / w.sum()), 3)}
    json.dump({"summary": summ, "maps": report}, open(f"{MDIR}/physics_report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    print(f"地图物理已保存到 {MDIR}，用时 {time.time() - t0:.0f}s")
    build_height_main()
    duel_stats_main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "contact":
        real_contact_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "duel":
        duel_stats_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "height":
        build_height_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "evalgen":
        eval_generation_main()
    else:
        print(__doc__)
