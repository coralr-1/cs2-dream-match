#!/bin/bash
# 安装后每 30 分钟检查一次「收件箱」（运行环境里的 inbox），发现拷贝完成的新数据就自动清洗、提取、重训模型。
R="${CS2SIM:-$HOME/cs2sim}"
P="$HOME/Library/LaunchAgents/com.cs2sim.autoupdate.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$R/logs"
sed "s#__CS2SIM__#$R#g" "$(dirname "$0")/tools/com.cs2sim.autoupdate.plist" > "$P"
launchctl bootout gui/$(id -u) "$P" 2>/dev/null
launchctl bootstrap gui/$(id -u) "$P" && echo "已安装自动更新（每 30 分钟检查一次收件箱）。日志：$R/data/update_log.txt"
