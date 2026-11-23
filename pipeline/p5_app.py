"""P5 网页界面：梦幻对决模拟 + 选手画像 + 战队画像。

运行：python p5_app.py            （默认监听 0.0.0.0:6006；本机使用时设 HOST=127.0.0.1 PORT=7860）
在云服务器上运行时，用 SSH 端口转发到本机再用浏览器打开。
"""
import paths
import os
import threading
from collections import Counter

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import p4_model as sim
import p12_veto as vt

C = sim.C
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Heiti SC", "Arial Unicode MS", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
VIDEO_DIR = paths.VIDEOS
plt.rcParams["axes.unicode_minus"] = False

S = sim.Simulator.load()
VETO = vt.Veto.load() if os.path.exists(f"{vt.MDIR}/veto_model.pt") else None
POOL_LABELS = [p["label"] for p in VETO.presets] if VETO else ["（还没有 Ban/Pick 模型）"]
maps = pd.read_parquet(f"{C}/maps.parquet")
players = pd.read_parquet(f"{C}/players.parquet")
roles = pd.read_parquet(f"{C}/player_roles.parquet")
py = pd.read_parquet(f"{C}/player_years.parquet")
places = pd.read_parquet(f"{C}/player_places.parquet")
habits = pd.read_parquet(f"{C}/team_habits.parquet")
centers = pd.read_parquet(f"{C}/place_centers.parquet")
POS = pads.dataset(f"{C}/positions.parquet") if os.path.exists(f"{C}/positions.parquet") else None   # 站位数据（热力图用）可能还没下载
NAME = dict(zip(players.steamid, players.name))
MAP_CN = {"de_ancient": "Ancient", "de_anubis": "Anubis", "de_cache": "Cache", "de_dust2": "Dust2", "de_inferno": "Inferno",
          "de_mirage": "Mirage", "de_nuke": "Nuke", "de_overpass": "Overpass", "de_train": "Train", "de_vertigo": "Vertigo"}
MAP_BY_CN = {v: k for k, v in MAP_CN.items()}

# ---------------------------------------------------------------- 选项
py_opts = py[py.rounds >= 100].merge(roles[["steamid", "year", "role"]], on=["steamid", "year"], how="left")
py_opts["label"] = py_opts.apply(lambda r: f"{r['name']} · {int(r.year)} · {r.role if isinstance(r.role, str) else '—'} · {int(r.rounds)}回合", axis=1)
py_opts = py_opts.sort_values(["name", "year"])
LABEL2PY = {r.label: (r.steamid, int(r.year)) for r in py_opts.itertuples()}
PLAYER_LABELS = list(LABEL2PY)


def team_rosters():
    rows = []
    for (slug, year), g in pd.concat([maps[["team1_slug", "year", "roster1"]].set_axis(["slug", "year", "roster"], axis=1),
                                      maps[["team2_slug", "year", "roster2"]].set_axis(["slug", "year", "roster"], axis=1)]).dropna().groupby(["slug", "year"]):
        if len(g) < 15:
            continue
        c = Counter(s for r in g.roster for s in str(r).split(","))
        rows.append({"slug": slug, "year": int(year), "maps": len(g), "roster": [s for s, _ in c.most_common(5)]})
    return pd.DataFrame(rows)


TEAMS = team_rosters()
TEAMS["label"] = TEAMS.apply(lambda r: f"{r.slug} · {r.year} · {r.maps}张图", axis=1)
TEAMS = TEAMS.sort_values(["slug", "year"])
TEAM_LABELS = list(TEAMS.label)


def label_of(sid, year):
    for lab, v in LABEL2PY.items():
        if v == (sid, year):
            return lab
    return None


def fill_team(team_label):
    if not team_label:
        return [None] * 5
    r = TEAMS[TEAMS.label == team_label].iloc[0]
    return [label_of(s, r.year) for s in r.roster] + [None] * (5 - len(r.roster))


# ---------------------------------------------------------------- 对决
def warn_for(team):
    w = []
    for sid, y in team:
        if S.vocab.ip(sid) == 0:
            w.append(f"{NAME.get(sid, sid)}：训练数据不足，按替补水平计算")
        elif S.vocab.ipy(sid, y) == 0:
            w.append(f"{NAME.get(sid, sid)}：{y} 年数据不足，使用他的通用水平")
    return w


def played_maps(team):
    yrs = {y for _, y in team}
    return set(maps[maps.year.isin(yrs)].map_name.unique())


def check_teams(A, B):
    """阵容合法返回 None，否则返回给用户看的提示。同一人可以用不同年份分别出现在两边（比如 25 年对 26 年的自己）。"""
    for t, team in (("A", A), ("B", B)):
        if len(team) != 5:
            return (f"{t} 队还有 {5 - len(team)} 个位置没选。（选手当年出场太少时不会被自动带出，请在空位里手动选一名）")
        if len({s for s, _ in team}) < 5:
            return f"{t} 队里同一名选手出现了两次。"
    both = set(A) & set(B)
    if both:
        s, y = next(iter(both))
        return (f"{NAME.get(s, s)}（{y}）同时出现在两边。同一名选手可以用不同年份分别出战（比如 2025 对 2026），"
                f"但同一年份只能在一边。")
    return None


def fail(msg, n_out):
    return (f"### ⚠️ {msg}",) + tuple(gr.update(visible=False) for _ in range(n_out - 1))


def run_match(a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, map_cn, n, progress=gr.Progress()):
    A = [LABEL2PY[x] for x in (a1, a2, a3, a4, a5) if x]
    B = [LABEL2PY[x] for x in (b1, b2, b3, b4, b5) if x]
    err = check_teams(A, B)
    if err:
        return fail(err, 3)
    nameA = " / ".join(f"{NAME.get(s, s)}({y})" for s, y in A)
    nameB = " / ".join(f"{NAME.get(s, s)}({y})" for s, y in B)
    common = sorted(played_maps(A) & played_maps(B))
    targets = common if map_cn == "全部共同地图" else [MAP_BY_CN[map_cn]]
    rows, dist = [], None
    for k, mp in enumerate(targets):
        progress(k / len(targets), desc=f"模拟 {MAP_CN[mp]}（{k + 1}/{len(targets)}）")
        out = S.simulate(A, B, mp, n=int(n))
        rows.append({"地图": MAP_CN[mp], "A 胜率": f"{out['p_win_A']:.1%}", "平均比分": f"{out['avg_rounds_A']:.1f} : {out['avg_rounds_B']:.1f}",
                     "加时概率": f"{out['p_overtime']:.1%}", "_p": out["p_win_A"], "_scores": out["top_scores"]})
    table = pd.DataFrame(rows).sort_values("_p", ascending=False)
    warns = warn_for(A) + warn_for(B)
    note = ""
    if map_cn != "全部共同地图" and MAP_BY_CN[map_cn] not in common:
        note = "\n\n⚠️ 至少一方在所选年份没有这张图的比赛数据，结果是外推的，仅供娱乐。"
    best = table.iloc[0]
    md = f"### A 队：{nameA}\n### B 队：{nameB}\n\n"
    if len(table) > 1:
        avg = table._p.mean()
        md += f"**各图平均 A 胜率 {avg:.1%}**；A 最强图 **{best['地图']}**（{best['A 胜率']}），最弱图 **{table.iloc[-1]['地图']}**（{table.iloc[-1]['A 胜率']}）。\n"
    else:
        md += f"**{best['地图']}：A 胜率 {best['A 胜率']}**，平均比分 {best['平均比分']}，加时概率 {best['加时概率']}。\n"
    if warns:
        md += "\n" + "\n".join(f"- {w}" for w in warns)
    md += note + "\n\n> 每名选手按他在所选年份的状态参战；跨年份对决是根据数据的推测，不代表真实比赛结果。"
    sc = table.iloc[0]._scores
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(sc.keys()), list(sc.values()), color="#4C78A8")
    ax.set_title(f"{best['地图']} 最可能的比分（A : B）")
    ax.set_ylabel("概率")
    fig.tight_layout()
    return md, gr.update(value=table.drop(columns=["_p", "_scores"]), visible=True), gr.update(value=fig, visible=True)


# ---------------------------------------------------------------- Ban/Pick
def team_display(team_label, team, side):
    tid = team_id_for(team_label, team)
    if not tid:
        return f"{side} 队（自选）"
    from render_video import team_name          # 队名写法（NAVI、The MongolZ……）和视频里保持一致
    return f"{team_name(tid[0])} {tid[1]}"


FIRST = {"随机": None, "A 先手": "A", "B 先手": "B"}


def map_win_probs(A, B, pool, progress, p0=0.0, p1=0.6):
    wp = {}
    for i, m in enumerate(pool):
        progress(p0 + (p1 - p0) * i / len(pool), desc=f"AI 估算每张图的胜率：{MAP_CN[m]}")
        wp[m] = float(S.simulate(A, B, m, n=1000)["p_win_A"])
    return wp


def run_veto(ta_label, tb_label, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, fmt, pool_label, first, seed, progress=gr.Progress()):
    A = [LABEL2PY[x] for x in (a1, a2, a3, a4, a5) if x]
    B = [LABEL2PY[x] for x in (b1, b2, b3, b4, b5) if x]
    err = check_teams(A, B) or (None if VETO else "还没有 Ban/Pick 模型。")
    if err:
        return fail(err, 2)
    pool = VETO.preset(pool_label)["pool"]
    wp = map_win_probs(A, B, pool, progress)
    progress(0.8, desc="模拟 Ban/Pick")
    v = VETO.simulate(A, B, fmt.lower(), pool_label, FIRST[first], int(seed), map_wp=wp)
    na, nb = team_display(ta_label, A, "A"), team_display(tb_label, B, "B")
    nm = {"A": na, "B": nb}
    lines, k = [], 0
    for st in v["flow"]:
        m = MAP_CN[st["map"]]
        if st["action"] == "ban":
            lines.append(f"- {nm[st['team']]} **禁** {m}　<small>（这一步的概率 {st['prob']:.0%}）</small>")
        elif st["action"] == "pick":
            k += 1
            lines.append(f"- {nm[st['team']]} **选** {m} → **第 {k} 图**，A 胜率 {wp[st['map']]:.0%}　<small>（概率 {st['prob']:.0%}）</small>")
        else:
            lab = "比赛地图" if v["fmt"] == "bo1" else "决胜图"
            lines.append(f"- 剩下 **{m}** → **{lab}**，A 胜率 {wp[st['map']]:.0%}")
    md = (f"### {fmt} Ban/Pick（剧本 {int(seed)}，{nm[v['first']]} 先手）\n" + "\n".join(lines) +
          f"\n\n**这套流程下 A 赢下系列赛的概率 {v['series_wp_flow']:.0%}**；把 ban/pick 的随机性也算进去，A 的系列赛胜率 **{v['series_wp']:.0%}**。"
          f"\n\n> Ban/Pick 由模型按两队选手在真实系列赛中的选图习惯模拟（数据里没有 ban 记录，模型从打图顺序反推）；"
          f"换「剧本编号」会得到另一种可能的流程。地图池：{v['preset']}")
    rows = []
    for m in pool:
        stt = v["stat"][m]
        rows.append({"地图": MAP_CN[m], "A 胜率": f"{wp[m]:.0%}", "A 禁": f"{stt['A禁']:.0%}", "B 禁": f"{stt['B禁']:.0%}",
                     "A 选": f"{stt['A选']:.0%}", "B 选": f"{stt['B选']:.0%}", "成为决胜/比赛图": f"{stt['决胜']:.0%}",
                     "_k": stt["A选"] + stt["B选"] + stt["决胜"]})
    tab = pd.DataFrame(rows).sort_values("_k", ascending=False).drop(columns="_k")
    return md, gr.update(value=tab, visible=True, label="每张图在 4000 次模拟里的去向")


# ---------------------------------------------------------------- 选手画像
def heatmap(sid, year, mp, side, t):
    if POS is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "站位数据还没下载完，热力图暂不可用", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    tbl = POS.to_table(filter=(pads.field("map_name") == mp) & (pads.field("t") == t) & (pads.field("alive")),
                       columns=["steamid", "year", "side", "X", "Y"]).to_pandas()
    fig, ax = plt.subplots(figsize=(6, 6))
    bg = tbl.sample(min(len(tbl), 40000), random_state=0)
    ax.scatter(bg.X, bg.Y, s=1, c="#dddddd", alpha=0.5, linewidths=0)
    me = tbl[(tbl.steamid == sid) & (tbl.year == year) & (tbl.side == side)]
    if len(me):
        ax.hexbin(me.X, me.Y, gridsize=45, cmap="inferno_r", mincnt=1, alpha=0.85)
    for c in centers[centers.map_name == mp].itertuples():
        ax.text(c.X, c.Y, c.place, fontsize=6, ha="center", color="#333333")
    ax.set_title(f"{NAME.get(sid, sid)} {year} · {MAP_CN[mp]} {side.upper()} · 冻结结束后第 {t} 秒（{len(me)} 个样本）")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def player_profile(label, map_cn, side, t):
    if not label:
        return "请选择选手。", None, None
    sid, year = LABEL2PY[label]
    r = py[(py.steamid == sid) & (py.year == year)].iloc[0]
    ro = roles[(roles.steamid == sid) & (roles.year == year)]
    role = ro.role.iat[0] if len(ro) else "（回合数不足，未分类）"
    lurk = ro.lurk_dist.iat[0] if len(ro) else np.nan
    md = (f"## {r['name']} · {year} · {role}\n\n"
          f"{int(r.maps)} 张图 / {int(r.rounds)} 回合 · 近似 Rating **{r.rating_approx:.2f}**\n\n"
          f"| 指标 | 全部 | CT | T |\n|---|---|---|---|\n"
          f"| 每回合击杀 | {r.kpr:.2f} | {r.ct_kpr:.2f} | {r.t_kpr:.2f} |\n"
          f"| ADR | {r.adr:.1f} | {r.ct_adr:.1f} | {r.t_adr:.1f} |\n"
          f"| KAST | {r.kast_pct:.1%} | {r.ct_kast_pct:.1%} | {r.t_kast_pct:.1%} |\n"
          f"| 首杀尝试/回合 | {r.opening_attempts_pr:.2f} | {r.ct_opening_attempts_pr:.2f} | {r.t_opening_attempts_pr:.2f} |\n\n"
          f"首杀成功率 {r.opening_success:.1%} · 爆头率 {r.hs_pct:.1%} · AWP 击杀占比 {r.awp_kill_share:.1%} · "
          f"每回合闪光 {r.flashes_pr:.2f} / 闪光助攻 {r.flash_assists_pr:.3f} / 道具伤害 {r.util_dmg_pr:.1f} · "
          f"补枪 {r.trade_kill_pr:.3f} / 被补 {r.traded_pr:.3f} · 存活率 {r.survival_pct:.1%} · "
          f"残局 {int(r.clutch_wins)}/{int(r.clutches)} · 首杀时间中位数 {r.first_kill_time:.0f}s"
          + (f" · T 方离队友平均 {lurk:.0f} 单位" if not np.isnan(lurk) else ""))
    pl = places[(places.steamid == sid) & (places.year == year) & (places.t == int(t))]
    tab = pl.assign(地图=pl.map_name.map(MAP_CN), 阵营=pl.side.str.upper(), 点位=pl.place, 占比=(pl.share * 100).round(0).astype(int).astype(str) + "%",
                    回合=pl.rounds)[["地图", "阵营", "点位", "占比", "回合"]].sort_values(["回合", "地图", "阵营"], ascending=[False, True, True])
    fig = heatmap(sid, year, MAP_BY_CN[map_cn], side.lower(), int(t))
    return md, tab, fig


# ---------------------------------------------------------------- 战队画像
def team_profile(team_label):
    if not team_label:
        return "请选择战队。", None
    r = TEAMS[TEAMS.label == team_label].iloc[0]
    roster = "、".join(f"{NAME.get(s, s)}（{(roles[(roles.steamid == s) & (roles.year == r.year)].role.tolist() or ['—'])[0]}）" for s in r.roster)
    h = habits[(habits.team_slug == r.slug) & (habits.year == r.year) & (habits.t_rounds >= 30)].copy()
    md = f"## {r.slug} · {r.year}\n\n主力阵容：{roster}\n\n共 {r.maps} 张图。下表按 T 方回合数排序。"
    pct = lambda s: (s * 100).round(0).map(lambda x: "—" if pd.isna(x) else f"{int(x)}%")
    tab = pd.DataFrame({
        "地图": h.map_name.map(MAP_CN), "T 回合": h.t_rounds.astype(int), "CT 胜率": pct(h.ct_win_pct), "T 胜率": pct(h.t_win_pct),
        "下包率": pct(h.plant_pct), "A/B 点": pct(h.site_a_pct) + " / " + pct(h.site_b_pct), "下包时间": h.plant_time_median.round(0).astype("Int64").astype(str) + "s",
        "手枪局": pct(h.pistol_win_pct), "经济局": pct(h.eco_win_pct), "强起": pct(h.force_win_pct),
        "CT 最常用站位(第15秒)": h.ct_top_setup.fillna("—") + "（" + pct(h.ct_top_setup_share) + "）"}).sort_values("T 回合", ascending=False)
    return md, tab


# ---------------------------------------------------------------- 生成短视频
_VA = {}
_VA_LOCK = threading.Lock()


def video_assets():
    with _VA_LOCK:                         # 启动时后台预加载；用户提前点按钮会在这里等加载完
        if not _VA:
            import render_video as rv
            _VA["rv"], _VA["assets"] = rv, rv.Assets()
    return _VA["rv"], _VA["assets"]


def team_id_for(team_label, team):
    """所选 5 人正好是该战队当年主力时才带上战队身份，否则按自由组合阵容处理。"""
    if not team_label:
        return None
    r = TEAMS[TEAMS.label == team_label].iloc[0]
    return (r.slug, int(r.year)) if {s for s, _ in team} == set(r.roster) and all(y == r.year for _, y in team) else None


VIDEO_KINDS = ["单张地图", "BO1（含 Ban/Pick）", "BO3（含 Ban/Pick，全程）", "选手高光（镜头跟随一名选手）"]


def hl_choices(*sel):
    opts = [x for x in sel if x]
    return gr.update(choices=opts, value=opts[0] if opts else None)


def make_short_video(ta_label, tb_label, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, kind, map_cn, pool_label, first, seed, dramatic, hl_label,
                     progress=gr.Progress()):
    A = [LABEL2PY[x] for x in (a1, a2, a3, a4, a5) if x]
    B = [LABEL2PY[x] for x in (b1, b2, b3, b4, b5) if x]
    err = check_teams(A, B) or (None if kind == VIDEO_KINDS[0] or VETO else "还没有 Ban/Pick 模型，只能生成单张地图。")
    if err:
        return gr.update(value=None), f"### ⚠️ {err}"
    progress(0.0, desc="加载模型（第一次较慢）")
    rv, assets = video_assets()
    ids = (team_id_for(ta_label, A), team_id_for(tb_label, B))
    nm = lambda ids_, team: f"{ids_[0]}{ids_[1]}" if ids_ else "自选" + NAME.get(team[0][0], "")
    os.makedirs(VIDEO_DIR, exist_ok=True)
    story = "dramatic" if dramatic else "typical"
    if kind == VIDEO_KINDS[3]:
        if not hl_label or hl_label not in (a1, a2, a3, a4, a5, b1, b2, b3, b4, b5):
            return gr.update(value=None), "### ⚠️ 请在「高光选手」里选一名场上的选手。"
        focus = LABEL2PY[hl_label]
        out = os.path.join(VIDEO_DIR, f"{nm(ids[0], A)}_vs_{nm(ids[1], B)}_{NAME.get(focus[0], '')}{focus[1]}高光_{MAP_BY_CN[map_cn].replace('de_', '')}_s{int(seed)}.mp4")
        _, q, keys = rv.make_highlight_video(assets, A, B, MAP_BY_CN[map_cn], out, focus, ids, n_sims=400, seed=int(seed),
                                             progress=lambda f, d: progress(f, desc=d))
        md = (f"**{q['name']} 高光**：本场 {q['kills']}-{q['deaths']}，高光回合 {', '.join(map(str, keys))}。\n\n"
              f"视频已保存到：`{out}`。换一个「剧本编号」可以得到另一场。")
        return gr.update(value=out), md
    if kind != VIDEO_KINDS[0]:
        fmt = "bo1" if kind.startswith("BO1") else "bo3"
        out = os.path.join(VIDEO_DIR, f"{nm(ids[0], A)}_vs_{nm(ids[1], B)}_{fmt}_s{int(seed)}.mp4")
        _, v, results = rv.make_series_video(assets, A, B, fmt, out, ids, n_sims=400, seed=int(seed), story=story, preset=pool_label,
                                             first=FIRST[first], progress=lambda f, d: progress(f, desc=d))
        sa = sum(r[3] == "A" for r in results)
        sb = len(results) - sa
        wname = team_display(ta_label, A, "A") if sa > sb else team_display(tb_label, B, "B")
        md = (f"**{wname} {max(sa, sb)} : {min(sa, sb)} 赢下 {fmt.upper()}**　" + "　".join(f"{MAP_CN[m]} {a}:{b}" for m, a, b, _ in results) +
              f"\n\nBan/Pick：" + " → ".join(MAP_CN[m] for m in v["order"]) +
              f"\n\n视频已保存到：`{out}`（同名 .png 是封面，.json 是 Ban/Pick 和逐图剧本）。换一个「剧本编号」可以得到另一种流程和结果。")
        return gr.update(value=out), md
    out = os.path.join(VIDEO_DIR, f"{nm(ids[0], A)}_vs_{nm(ids[1], B)}_{MAP_BY_CN[map_cn].replace('de_', '')}_s{int(seed)}.mp4")
    rv.make_video(assets, A, B, MAP_BY_CN[map_cn], out, ids, n_sims=400, seed=int(seed), story=story,
                  progress=lambda f, d: progress(f, desc=d))
    import json as _json
    st = _json.load(open(out.rsplit(".", 1)[0] + ".json"))
    fa, fb = st["story"]["final"]
    mvp = next(p for p in st["story"]["players"] if p["slot"] == st["story"]["mvp"])
    md = (f"**剧本比分 {fa} : {fb}**，MVP {mvp['name']}（{mvp['kills']}-{mvp['deaths']}）；赛前模拟 A 胜率 {st['p_win_A']:.0%}。\n\n"
          f"视频已保存到：`{out}`（同名 .png 是封面，.json 是逐回合剧本）。换一个「剧本编号」可以得到另一场比赛。")
    return gr.update(value=out), md


# ---------------------------------------------------------------- 界面
with gr.Blocks(title="CS2 梦幻对决") as app:
    gr.Markdown("# CS2 梦幻对决模拟器\n基于 2023–2026 年 S 级赛事 2298 场比赛、11 万回合的数据。")
    with gr.Tab("梦幻对决"):
        with gr.Row():
            with gr.Column():
                ta = gr.Dropdown(TEAM_LABELS, label="A 队：快速选择战队-年份（自动带出主力）")
                A = [gr.Dropdown(PLAYER_LABELS, label=f"A{i + 1}", filterable=True) for i in range(5)]
            with gr.Column():
                tb = gr.Dropdown(TEAM_LABELS, label="B 队：快速选择战队-年份（自动带出主力）")
                B = [gr.Dropdown(PLAYER_LABELS, label=f"B{i + 1}", filterable=True) for i in range(5)]
        ta.change(fill_team, ta, A)
        tb.change(fill_team, tb, B)
        with gr.Row():
            mp = gr.Dropdown(["全部共同地图"] + list(MAP_CN.values()), value="全部共同地图", label="地图")
            n = gr.Slider(500, 5000, value=2000, step=500, label="模拟场数")
            go = gr.Button("开始模拟", variant="primary")
        out_md = gr.Markdown()
        out_tab = gr.Dataframe(visible=False)
        out_fig = gr.Plot(visible=False)
        go.click(run_match, A + B + [mp, n], [out_md, out_tab, out_fig])
        gr.Markdown("---\n### Ban/Pick 模拟（BO1 / BO3 / BO5）")
        with gr.Row():
            bfmt = gr.Radio(["BO1", "BO3", "BO5"], value="BO3", label="赛制")
            bfirst = gr.Radio(list(FIRST), value="随机", label="谁先 ban")
            bseed = gr.Number(value=0, precision=0, label="剧本编号（换数字看另一种流程）")
        with gr.Row():
            bpool = gr.Dropdown(POOL_LABELS, value=POOL_LABELS[0], label="地图池（默认最新）", scale=3)
            bbtn = gr.Button("模拟 Ban/Pick", variant="primary", scale=1)
        b_md = gr.Markdown()
        b_tab = gr.Dataframe(visible=False)
        bbtn.click(run_veto, [ta, tb] + A + B + [bfmt, bpool, bfirst, bseed], [b_md, b_tab])
        gr.Markdown("---\n### 生成短视频（9:16 竖屏，2D 小地图对战过程）")
        with gr.Row():
            vkind = gr.Radio(VIDEO_KINDS, value=VIDEO_KINDS[0], label="视频内容")
            vmap = gr.Dropdown(list(MAP_CN.values()), value="Mirage", label="地图（「单张地图」「选手高光」时用）")
            vhl = gr.Dropdown([], label="高光选手（「选手高光」时用，自动列出场上 10 人）")
        with gr.Row():
            vseed = gr.Number(value=0, precision=0, label="剧本编号（换数字得到另一场）")
            vdram = gr.Checkbox(value=False, label="挑胶着的（单图打到加时；BO3 打满三图）")
            vbtn = gr.Button("生成短视频", variant="primary")
        gr.Markdown("用上面选好的两队。BO1/BO3 会先播 Ban/Pick（地图池和先手用上面「Ban/Pick 模拟」里的设置），再逐图打完。"
                    "同一名选手可以用不同年份分别出现在两边。单图约 1–2 分钟，BO3 约 4–6 分钟，下方会显示进度条。")
        v_md = gr.Markdown()
        v_out = gr.Video(label="视频（生成时这里显示进度条）", height=640)
        vbtn.click(make_short_video, [ta, tb] + A + B + [vkind, vmap, bpool, bfirst, vseed, vdram, vhl], [v_out, v_md], show_progress="full")
        for dd in A + B:
            dd.change(hl_choices, A + B, vhl)
    with gr.Tab("选手画像"):
        with gr.Row():
            pl = gr.Dropdown(PLAYER_LABELS, label="选手 · 年份", filterable=True)
            pm = gr.Dropdown(list(MAP_CN.values()), value="Mirage", label="热力图地图")
            ps = gr.Radio(["CT", "T"], value="CT", label="阵营")
            pt = gr.Radio([15, 30, 55], value=30, label="冻结结束后第几秒")
        pbtn = gr.Button("查看", variant="primary")
        p_md = gr.Markdown()
        with gr.Row():
            p_tab = gr.Dataframe(label="常站点位（前 3）")
            p_fig = gr.Plot()
        pbtn.click(player_profile, [pl, pm, ps, pt], [p_md, p_tab, p_fig])
    with gr.Tab("战队画像"):
        tsel = gr.Dropdown(TEAM_LABELS, label="战队 · 年份")
        tbtn = gr.Button("查看", variant="primary")
        t_md = gr.Markdown()
        t_tab = gr.Dataframe()
        tbtn.click(team_profile, tsel, [t_md, t_tab])

if __name__ == "__main__":
    threading.Thread(target=video_assets, daemon=True).start()       # 后台先把视频模型加载好，第一次出片更快
    os.makedirs(VIDEO_DIR, exist_ok=True)
    # 视频保存在 VIDEO_DIR（不在程序目录下），必须加进 allowed_paths，否则 Gradio 拒绝播放生成好的视频（页面上只显示 Error）
    app.launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 6006)),
               allowed_paths=[VIDEO_DIR], show_error=True)
