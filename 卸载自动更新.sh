#!/bin/bash
P="$HOME/Library/LaunchAgents/com.cs2sim.autoupdate.plist"
launchctl bootout gui/$(id -u) "$P" 2>/dev/null; rm -f "$P" && echo "已卸载自动更新"
