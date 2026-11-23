#!/bin/bash
# Mac 一键准备运行环境（默认 ~/cs2sim，可用 CS2SIM=/其他目录 覆盖）。
#   ./setup_mac.sh          首次安装：建目录、建 Python 环境、装依赖、拷代码、建快捷方式
#   ./setup_mac.sh --sync   只同步代码（改了 pipeline/ 之后运行）
# 为什么不直接在项目目录里跑：macOS 不允许后台定时任务访问「桌面」「文稿」里的文件，自动更新需要运行环境放在别处。
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
R="${CS2SIM:-$HOME/cs2sim}"
mkdir -p "$R/pipeline" "$R/data/clean" "$R/data/models" "$R/inbox" "$R/videos" "$R/logs"
rsync -a --delete --exclude "__pycache__" "$REPO/pipeline/" "$R/pipeline/"
cat > "$R/auto_update.sh" <<EOS
#!/bin/bash
# 自动更新入口：由定时任务每 30 分钟调用一次；inbox 里没有新数据时什么也不做。手动运行：bash $R/auto_update.sh --force
R="$R"
export CS2_HOME="\$R/data" CS2_INBOX="\$R/inbox" CS2_CLEAN="\$R/data/clean" CS2_MODELS="\$R/data/models" PYTORCH_ENABLE_MPS_FALLBACK=1
cd "\$R" && "\$R/.venv/bin/python" "\$R/pipeline/auto_update.py" "\$@" >> "\$R/logs/auto_update.out" 2>&1
EOS
chmod +x "$R/auto_update.sh"
[ -e "$REPO/收件箱" ] || ln -s "$R/inbox" "$REPO/收件箱"
[ -e "$REPO/生成的视频" ] || ln -s "$R/videos" "$REPO/生成的视频"
if [ "$1" = "--sync" ]; then echo "代码已同步到 $R/pipeline"; exit 0; fi
if [ ! -x "$R/.venv/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv --python 3.12 "$R/.venv" && uv pip install --python "$R/.venv/bin/python" -r "$REPO/requirements.txt"
  else python3 -m venv "$R/.venv" && "$R/.venv/bin/pip" install -r "$REPO/requirements.txt"; fi
fi
echo "运行环境已就绪：$R"
echo "接下来：把清洗好的数据放进 $R/data/clean、训练好的模型放进 $R/data/models（训练方法见 README），然后双击「打开模拟器.command」。"
