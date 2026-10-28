# -*- coding: utf-8 -*-
"""
d3_split_denoise_loudness.py —— 第 3 天·静音切分 + 降噪 + 响度归一
================================================================
三个任务，一个脚本：

1) VAD 切分（语音活动检测）
   原理（高数上册）：把音频按 30ms 一帧，算每帧短时能量
       E_n = (1/N)·Σ x_i²
   能量 > 阈值 → 有人在说话；连续的能量低谷 → 静音，在这里切开。
   比喻：VAD 就是声音版的人体感应灯——有动静就"亮"。

2) 降噪（谱减法, spectral subtraction）
   模型：y(t) = s(t) + n(t)，我们只录到了 y。
   假设噪声是"平稳的"（底噪不随时间突变），用静音段估计噪声谱 n̂(f)，
   从整段频谱里减掉：ŝ(f) = max(|y(f)| - α·n̂(f), 0)
   ⚠️ 重难点：过度降噪会把语音特征一起抹掉——擦橡皮太用力，纸也破了。

3) 响度归一化
   给每条音频乘一个标量 c 使峰值/能量达到统一标准：x → c·x（线代的数乘）。
   为什么必须做：不让模型把"音量大"当成特征，逼它只学内容。

运行：python scripts/d3_split_denoise_loudness.py
输出：output/segments/*.wav（切分后的语段）+ reports/D3_切分降噪报告.md
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def frame_energy(x: np.ndarray, frame_len: int) -> np.ndarray:
    """把向量切帧，逐帧算能量 → 返回能量向量。
    线代视角： reshape 成 (n_frames, frame_len) 矩阵后对每行算平方均值，
    等价于能量向量 = row-wise mean(X²)，一次矩阵运算搞定。"""
    n = len(x) // frame_len
    frames = x[: n * frame_len].reshape(n, frame_len)
    return np.mean(frames ** 2, axis=1)


def vad_split(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """基于短时能量阈值切分。返回 [(start_sample, end_sample), ...]"""
    frame_len = sr * config.FRAME_MS // 1000
    energies = frame_energy(x, frame_len)

    # 阈值自适应（关键修复 2026-10-27）：
    # 旧版固定阈值 0.02 对"轻声朗读"太高——最长连续语音只有 5 帧（差 1 帧不够 200ms），
    # 整条录音被误判成静音，切出 0 段（98 个文件中招）。
    # 新版：先估计本文件的"噪声底" = 能量 20% 分位（最安静的那 20% 帧），
    # 阈值 = max(保底下限, 噪声底 × 8)。比喻：空调噪音大的房间，说话要盖过 8 倍噪音才算"开口"；
    # 安静的房间（噪声底≈0），只要超过 0.004 的保底线就算。
    floor = np.percentile(energies, 20)
    thr = max(config.ENERGY_THRESHOLD, floor * 8)

    voiced = energies > thr  # 布尔向量：True=说话帧（线代的比较运算返回布尔矩阵）

    segments, start = [], None
    min_speech_frames = sr * config.MIN_SPEECH_MS // 1000 // frame_len
    pad_frames = sr * config.PAD_MS // 1000 // frame_len

    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_speech_frames:      # 太短的"咔哒声"丢弃
                s = max(0, (start - pad_frames) * frame_len)          # 前留缓冲
                e = min(len(x), (i + pad_frames) * frame_len)         # 后留缓冲
                segments.append((s, e))
            start = None
    if start is not None and len(voiced) - start >= min_speech_frames:
        s = max(0, (start - pad_frames) * frame_len)
        segments.append((s, len(x)))

    # 去重叠（修复 2026-10-27）：前后各补 100ms 缓冲会让相邻语段互相压住，
    # 实测「录音__望思台」seg002 结束 3.78s > seg003 开始 3.69s，重叠 90ms
    # ——同一段音频被算两次，训练集里等于给模型看重复样本。
    # 处理：两段相邻就把重叠区对半分，各自让一半。
    for k in range(len(segments) - 1):
        s1, e1 = segments[k]
        s2, e2 = segments[k + 1]
        if e1 > s2:                       # 有重叠
            mid = (s2 + e1) // 2          # 取重叠区中点切开
            segments[k] = (s1, mid)
            segments[k + 1] = (mid, e2)
    return segments


def spectral_denoise(x: np.ndarray, sr: int) -> np.ndarray:
    """谱减降噪：FFT → 减噪声谱 → 逆 FFT（傅里叶级数的实战应用）"""
    frame_len = 512
    hop = frame_len // 2
    # 分帧加窗（汉宁窗：两端归零，减少切帧边缘的频谱泄漏）
    n_frames = (len(x) - frame_len) // hop + 1
    if n_frames < 3:
        return x
    frames = np.stack([x[i * hop: i * hop + frame_len] * np.hanning(frame_len)
                       for i in range(n_frames)])

    spec = np.fft.rfft(frames, axis=1)                    # 时域 → 频域
    mag, phase = np.abs(spec), np.angle(spec)

    # 用能量最低的 20% 帧估计噪声谱（那些帧里基本只有底噪）
    frame_mag = np.linalg.norm(mag, axis=1)
    noise_idx = np.argsort(frame_mag)[: max(1, n_frames // 5)]
    noise_mag = mag[noise_idx].mean(axis=0)

    over_sub = 2.0  # 过减系数 α：宁可少减（保守），过度降噪会伤语音
    clean_mag = np.maximum(mag - over_sub * noise_mag, 0.01 * mag)  # 谱底防"音乐噪声"

    clean = np.fft.irfft(clean_mag * np.exp(1j * phase), n=frame_len, axis=1)  # 频域 → 时域
    out = np.zeros(n_frames * hop + frame_len)
    for i in range(n_frames):   # 重叠相加（overlap-add）
        out[i * hop: i * hop + frame_len] += clean[i]
    return out[: len(x)]


def normalize_loudness(x: np.ndarray) -> np.ndarray:
    """峰值归一到 -1dBFS：c·x，使 max|x| = 10^(-1/20) ≈ 0.891"""
    peak = np.abs(x).max()
    if peak < 1e-6:
        return x
    return x * (10 ** (-1 / 20) / peak)


def main():
    files = sorted(config.DATA_CLEAN.glob("*.wav"))
    if not files:
        print("data/clean 为空 —— 请先运行 d2_format_and_filter.py")
        sys.exit(1)

    seg_dir = config.OUTPUT / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    stats = []

    for p in files:
        x, sr = sf.read(p, dtype="float32")
        x_dn = spectral_denoise(x, sr)                 # ② 降噪
        x_norm = normalize_loudness(x_dn)              # ③ 响度归一
        segments = vad_split(x_norm, sr)               # ① 切分

        n_seg = 0
        for j, (s, e) in enumerate(segments):
            seg = x_norm[s:e]
            if len(seg) / sr < 0.3:                    # 切完仍 <0.3s 的丢掉
                continue
            out = seg_dir / f"{p.stem}_seg{j:03d}.wav"
            sf.write(out, seg.astype(np.float32), sr, subtype="PCM_16")
            n_seg += 1

        stats.append({"文件": p.name, "切出语段": n_seg,
                      "总时长s": round(len(x) / sr, 1),
                      "语段总时长s": round(sum(e - s for s, e in segments) / sr, 2)})

    lines = [
        "# D3 切分降噪报告\n",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
        f"- 输入：{len(files)} 条（降噪+归一+VAD 切分）",
        f"- 切出语段：{sum(s['切出语段'] for s in stats)} 段 → output/segments/\n",
        "| 文件 | 总时长(s) | 切出语段数 | 语段合计(s) |",
        "|---|---|---|---|",
    ]
    lines += [f"| {s['文件']} | {s['总时长s']} | {s['切出语段']} | {s['语段总时长s']} |"
              for s in stats]

    report = config.REPORTS / "D3_切分降噪报告.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (config.REPORTS / "D3_detail.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已保存：{report}")


if __name__ == "__main__":
    main()
