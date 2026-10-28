# -*- coding: utf-8 -*-
"""
config.py —— 全流水线的"总开关"配置文件
========================================
比喻：这是厨房的调料架。所有脚本都从这里取参数，
你想调"火候"（阈值、采样率）只需要改这一个文件，不用翻别的代码。

核心概念对照：
- TARGET_SR   采样率（每秒"拍照"次数）。奈奎斯特定理：语音信息主要在 4kHz 内，
              采样率 >= 2*4000 = 8000 就不丢信息，取 16000 是行业通用（留足余量）。
- TARGET_CH   声道数。单声道 = 一只耳朵听。ASR 训练都用单声道，双声道信息是冗余的。
- BIT_DEPTH   位深（量化档位）。16-bit = 2^16 = 65536 级振幅刻度。
"""

from pathlib import Path

# ---------- 路径 ----------
# 项目根目录 = 本文件所在目录的上一级（audio_pipeline/）
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"        # 你把原始音频放这里（mp3/wav/m4a 都行）
DATA_CLEAN = ROOT / "data" / "clean"    # 清洗后的 16kHz 单声道 wav
OUTPUT = ROOT / "output"                # 切分、转写、manifest 等最终产物
REPORTS = ROOT / "reports"              # 每个环节的清洗报告（多少条进/出/为什么）
DOCS = ROOT / "docs"                    # 讲义
TOOLS = ROOT / "tools"                  # ffmpeg 便携版放这里

# 自动寻找 ffmpeg（不需要你配置环境变量，脚本自己找）
def find_ffmpeg():
    """在 tools/ 目录里找 ffmpeg.exe；找不到返回 None（check_env 会提示）"""
    if not TOOLS.exists():
        return None
    for exe in TOOLS.rglob("ffmpeg.exe"):
        return str(exe)
    return None

# ---------- 音频参数 ----------
TARGET_SR = 16000      # 采样率：16kHz（行业标配，见上方注释）
TARGET_CH = 1          # 声道：单声道
BIT_DEPTH = "PCM_16"   # 16-bit 量化

# ---------- D2 质量过滤阈值（海关安检的"合格标准"）----------
MIN_DURATION = 1.0     # 短于 1 秒的音频：信息量太少，剔除
MAX_DURATION = 30.0    # 长于 30 秒：不利于训练切批，交给 D3 切分处理
CLIP_RATIO_MAX = 0.001 # 削波（爆音）采样点占比上限：超过说明录音顶格破音
SNR_MIN_DB = 10.0      # 信噪比下限：<10dB 的录音"噪声比人声还响"，对训练有害无益

# ---------- D3 VAD 切分参数 ----------
FRAME_MS = 30          # 每帧 30ms（约 480 个采样点 @16kHz），VAD 的最小判断单位
ENERGY_THRESHOLD = 0.004 # 短时能量阈值的"保底下限"（归一化后相对值）；
                         # 实际阈值 = max(保底下限, 本文件噪声底 × 8)，见 d3 脚本
MIN_SPEECH_MS = 200    # 语段最短 200ms：太短的"咔哒声"当噪声丢掉
PAD_MS = 100           # 每段前后各保留 100ms 缓冲，防止把音节首尾切掉

# ---------- D5 去重参数 ----------
SIMILARITY_THRESHOLD = 0.9  # 两段文本相似度 > 0.9 判定为重复（编辑距离归一化）
