#!/bin/bash
# 在云服务器上启动/重启网页：bash run_app.sh   （日志 p5.log，进程号 p5.pid）
# 数据目录用 CS2_DATA 指定，例如 CS2_DATA=/path/to/cs2 bash run_app.sh
cd "$(dirname "$0")"
[ -f p5.pid ] && kill $(cat p5.pid) 2>/dev/null
setsid nohup "${PYTHON:-python3}" p5_app.py > p5.log 2>&1 < /dev/null &
echo $! > p5.pid
