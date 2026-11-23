#!/bin/bash
# 双击打开「CS2 梦幻对决模拟器」网页（本机离线运行）。关掉这个终端窗口即退出。
R="${CS2SIM:-$HOME/cs2sim}"
export CS2_CLEAN="$R/data/clean" CS2_MODELS="$R/data/models" CS2_VIDEOS="$R/videos" PYTORCH_ENABLE_MPS_FALLBACK=1
export HOST=127.0.0.1 PORT=7860
cd "$R/pipeline"
echo "正在启动，稍后会自动打开浏览器 http://localhost:7860 ……"
( for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7860 && open http://localhost:7860 && break; sleep 2; done ) &
"$R/.venv/bin/python" p5_app.py
