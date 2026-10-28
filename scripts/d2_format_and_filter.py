# -*- coding: utf-8 -*-
"""
d2_format_and_filter.py —— 第 2 天·格式归一化 + 质量过滤
========================================================
两个任务：
1) 格式归一化：任何格式(mp3/m4a/flac...) → 16kHz 单声道 16-bit wav
   比喻：所有进口商品先换成"国标插头"，后面的机器才能统一处理。
2) 质量过滤（海关安检）：逐条检查，不合格的拦下并写明理由
   - 时长 < 1s          → 信息量太少
   - 时长 > 30s         → 不剔除，标记"待 D3 切分"
   - 削波（爆音）占比高  → 波形顶格截断，内容已失真
   - SNR < 10dB         → 噪声比人声响，对训练有害无益

学到的原理：
- SNR（信噪比）= 10·lg(信号功率/噪声功率)，单位 dB（对数尺度）
- 削波检测：|x| ≥ 0.99 的采样点占比
- 重采样：librosa.load(sr=16000) 内部做多项式插值，把任意采样率"拉"到 16kHz

运行：python scripts/d2_format_and_filter.py
输出：data/clean/*.wav  +  reports/D2_质量过滤报告.md
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# 尝试导入 librosa（重采样用它；若没装则退回 soundfile 的原生读取+线性插值）
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def decode_with_ffmpeg(src: Path, tmp_wav: Path) -> tuple[bool, str]:
    """用 ffmpeg 把任意格式解码成 16kHz/单声道/16-bit wav（一步完成格式归一）。

    为什么必须用它：libsndfile（soundfile 的底层）只认 wav/flac/ogg/mp3，
    **不认 m4a/aac**——而这次 760 个文件正是手机录音的 m4a/aac。
    用 librosa.load 读它们会全部抛 LibsndfileError，看起来像"数据坏了"，
    其实只是"钥匙不对"。ffmpeg 才是全格式的万能解码器。
    """
    ff = config.find_ffmpeg()
    if not ff:
        return False, "找不到 ffmpeg"
    cmd = [ff, "-v", "error", "-y", "-i", str(src),
           "-ar", str(config.TARGET_SR), "-ac", str(config.TARGET_CH),
           "-c:a", "pcm_s16le", str(tmp_wav)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="ignore")
    except subprocess.TimeoutExpired:
        return False, "ffmpeg 超时"
    if r.returncode != 0:
        return False, (r.stderr or "").strip()[:120]
    return True, ""


def probe_duration(path: Path) -> float | None:
    """用 ffprobe 查原始时长（不读进内存，秒回）。
    ffprobe 是 ffmpeg 自带的"体检仪"，只看文件元数据不碰音频内容。"""
    ffmpeg_dir = Path(config.find_ffmpeg()).parent if config.find_ffmpeg() else None
    ffprobe = ffmpeg_dir / "ffprobe.exe" if ffmpeg_dir else None
    if ffprobe and ffprobe.exists():
        try:
            out = subprocess.run(
                [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=30)
            return float(json.loads(out.stdout)["format"]["duration"])
        except Exception:
            return None
    return None


def estimate_snr_db(x: np.ndarray) -> float:
    """估计信噪比（不用模型的简易版）。

    思路：把音频按帧算短时能量 E_n = (1/N)·Σ x_i²（高数的黎曼和离散化），
    能量最高的 10% 帧当"信号"，最低的 10% 帧当"噪声"。
    SNR = 10·lg(P_signal / P_noise)
    """
    frame = config.TARGET_SR * config.FRAME_MS // 1000  # 每帧点数
    n_frames = len(x) // frame
    if n_frames < 4:
        return 99.0  # 太短，不做 SNR 判断
    energies = np.array([
        np.mean(x[i * frame:(i + 1) * frame] ** 2) for i in range(n_frames)
    ])
    energies = np.sort(energies)
    k = max(1, int(n_frames * 0.1))
    p_signal = energies[-k:].mean()   # 最响的 10% 帧
    p_noise = energies[:k].mean() + 1e-12  # 最静的 10% 帧（加小量防除零）
    return float(10 * np.log10(p_signal / p_noise))


def main():
    exts = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus"}
    files = sorted(p for p in config.DATA_RAW.rglob("*") if p.suffix.lower() in exts)
    config.DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    kept, dropped, to_split = [], [], []
    for p in files:
        out_name = p.stem + ".wav"
        out_path = config.DATA_CLEAN / out_name
        tmp_wav = config.DATA_CLEAN / ("~tmp_" + out_name)
        try:
            # ① 解码 + 重采样 + 转单声道：ffmpeg 一步到位，写成 16k/mono/pcm_s16le
            ok, why = decode_with_ffmpeg(p, tmp_wav)
            if not ok:
                dropped.append((p.name, f"ffmpeg 解码失败：{why}"))
                continue
            x, _ = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            dur = len(x) / config.TARGET_SR

            # ② 时长检查
            if dur < config.MIN_DURATION:
                tmp_wav.unlink(missing_ok=True)
                dropped.append((p.name, f"时长 {dur:.2f}s < {config.MIN_DURATION}s，信息量太少"))
                continue
            if dur > config.MAX_DURATION:
                to_split.append((p.name, f"时长 {dur:.1f}s > {config.MAX_DURATION}s，交给 D3 切分"))

            # ③ 削波检查：|x|>=0.99 的点占比
            clip_ratio = float(np.mean(np.abs(x) >= 0.99))
            if clip_ratio > config.CLIP_RATIO_MAX:
                tmp_wav.unlink(missing_ok=True)
                dropped.append((p.name, f"削波占比 {clip_ratio:.2%}（波形顶格截断，爆音）"))
                continue

            # ④ SNR 检查
            snr = estimate_snr_db(x)
            if snr < config.SNR_MIN_DB and dur <= config.MAX_DURATION:
                tmp_wav.unlink(missing_ok=True)
                dropped.append((p.name, f"SNR {snr:.1f}dB < {config.SNR_MIN_DB}dB（噪声盖过人声）"))
                continue

            # ⑤ 通过安检 → 临时文件转正（已是 16k/mono/16bit）
            tmp_wav.replace(out_path)
            entry = {"原始文件": p.name, "清洗后": out_name,
                     "时长s": round(dur, 2), "SNR_dB": round(snr, 1),
                     "削波占比": f"{clip_ratio:.4%}"}
            kept.append(entry)

        except Exception as e:
            tmp_wav.unlink(missing_ok=True)
            dropped.append((p.name, f"读取/解码失败：{type(e).__name__}: {e}"))

    # ---------- 写报告 ----------
    lines = [
        "# D2 质量过滤报告\n",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
        f"- 输入：{len(files)} 条",
        f"- ✅ 通过：{len(kept)} 条（已统一为 16kHz/单声道/16-bit，存入 data/clean）",
        f"- ₢ 过长待切分：{len(to_split)} 条（不剔除，D3 处理）",
        f"- ❌ 剔除：{len(dropped)} 条\n",
    ]
    if to_split:
        lines.append("## 待切分清单\n")
        for name, why in to_split:
            lines.append(f"- {name}：{why}")
    if dropped:
        lines.append("\n## 剔除清单（每条都有理由——'数据出了什么问题'比'丢了多少'更重要）\n")
        for name, why in dropped:
            lines.append(f"- {name}：{why}")
    if kept:
        lines.append("\n## 通过明细\n")
        lines.append("| 文件 | 时长(s) | SNR(dB) | 削波占比 |")
        lines.append("|---|---|---|---|")
        for e in kept:
            lines.append(f"| {e['清洗后']} | {e['时长s']} | {e['SNR_dB']} | {e['削波占比']} |")

    report = config.REPORTS / "D2_质量过滤报告.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    # 明细同时存 json，供 D5 汇总
    (config.REPORTS / "D2_detail.json").write_text(
        json.dumps({"kept": kept, "dropped": dropped, "to_split": to_split},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已保存：{report}")


if __name__ == "__main__":
    main()
