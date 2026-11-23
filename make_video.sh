#!/bin/bash
# 在 Mac 本地生成「AI 模拟对局」短视频（不需要服务器）。运行环境和模型在 ~/cs2sim/。
#
# 单条（战队:年份）：
#   ./make_video.sh --a vitality:2025 --b spirit:2026 --map de_mirage --out 生成的视频/vit25_vs_spi26.mp4
# 自由组合阵容（选手名:年份，逗号分隔 5 人）：
#   ./make_video.sh --a-players "ZywOo:2025,ropz:2025,flameZ:2025,mezii:2025,apEX:2025" \
#                   --b-players "donk:2026,sh1ro:2026,zont1x:2026,magixx:2026,tN1R:2026" --map de_inferno --out 生成的视频/x.mp4
# 批量（见 对阵清单示例.csv）：
#   ./make_video.sh --batch 对阵清单示例.csv --outdir 生成的视频/
# 其他参数：--sims 400（模拟场数）  --seed 0（换个数字会得到另一场剧本）  --story dramatic（挑一场打到加时、最胶着的）
# 含 Ban/Pick 的系列赛：--series bo1|bo3|bo5（忽略 --map）  --pool 0（地图池，0=最新）  --first A|B（谁先 ban，默认随机）
# 选手高光（从之前生成的对局视频里剪）：--highlight donk:2026 --from 生成的视频/x.mp4 [--part 1]
R="${CS2SIM:-$HOME/cs2sim}"
export CS2_CLEAN="$R/data/clean" CS2_MODELS="$R/data/models" PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$R/.venv/bin/python" "$R/pipeline/render_video.py" "$@"
