#!/bin/bash
# 用全部数据训练正式出片用的模型，输出到 $CS2_DATA/models_final（需要先跑完 p0–p9、p13、p14）
# 例如：CS2_DATA=/path/to/cs2 bash train_final.sh
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export CS2_MODELS="${CS2_DATA:?请设置 CS2_DATA}/models_final" CS2_FINAL=1 CS2_CALIB_SRC="$CS2_DATA/models/calib.json"
mkdir -p "$CS2_MODELS"
$PY p4_model.py train > final_p4.log 2>&1 && \
$PY p12_veto.py train > final_p12.log 2>&1 && \
CS2_TAG=team2 $PY p7_event_model.py train > final_p7.log 2>&1 && mv "$CS2_MODELS/event_team2" "$CS2_MODELS/event_team" && \
$PY p13_physics.py build > final_p13.log 2>&1 && \
$PY p15_utility_model.py train > final_p15.log 2>&1 && \
$PY p10_motion.py train > final_p10.log 2>&1
echo "FINAL_DONE $?" >> final_p10.log
