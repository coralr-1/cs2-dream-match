"""数据、模型、视频目录。默认都在仓库根目录的 data/ 下，可用环境变量覆盖：

  CS2_DATA    总目录（默认 <仓库>/data）；Kaggle 原始数据（cs2-s-tier-demos-*）直接放在这里
  CS2_ROOT    原始数据目录（默认 = CS2_DATA）
  CS2_CLEAN   清洗后的表（默认 CS2_DATA/clean）
  CS2_MODELS  模型（默认 CS2_DATA/models）
  CS2_VIDEOS  视频输出（默认 CS2_DATA/videos）
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("CS2_DATA", os.path.join(REPO, "data"))
ROOT = os.environ.get("CS2_ROOT", DATA)
CLEAN = os.environ.get("CS2_CLEAN", os.path.join(DATA, "clean"))
MODELS = os.environ.get("CS2_MODELS", os.path.join(DATA, "models"))
VIDEOS = os.environ.get("CS2_VIDEOS", os.path.join(DATA, "videos"))
