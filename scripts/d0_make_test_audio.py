# -*- coding: utf-8 -*-
"""
d0_make_test_audio.py —— 合成"假语音"，用于无真实数据时验证流水线
================================================================
为什么需要它：真实音频还没到位，但流水线今天就要验证跑得通。
我们用数学合成一段"像人说话"的音频：
  - 说话段 = 基频 F0 + 若干谐波 + 幅度包络（模拟元音的共振峰结构）
  - 静音段 = 低幅白噪声（模拟环境底噪）

这里全是你高数课见过的东西：
  sin(2πft)          —— 傅里叶级数的单项
  谐波叠加 Σ A_k sin(kωt) —— 傅里叶级数本身
  幅度包络 e^{-t/τ}  —— 指数衰减（元音的"音头响音尾弱"）

运行：python scripts/d0_make_test_audio.py
产物：data/raw/ 下生成 5 个测试 wav（其中 1 个故意超长、1 个几乎全静音，
      用来检验 D2 质量过滤和 D3 切分是否真的起作用）
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def synth_speech_like(sr: int, duration: float, f0: float = 120.0,
                      seed: int = 0) -> np.ndarray:
    """合成一段'像说话'的波形。

    sr       : 采样率（每秒 sr 个点）
    duration : 秒数
    f0       : 基频 120Hz ≈ 成年男性嗓音（女声约 210Hz）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr

    # 傅里叶级数：基频 + 5 个谐波，高次谐波幅度按 1/k 衰减
    # （真实元音的频谱就是这样一串"梳齿"——谐波峰）
    wave = np.zeros_like(t)
    for k in range(1, 6):
        wave += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))

    # 幅度包络：用低频噪声调制，模拟音节强弱起伏（说话不是匀速的）
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6)))
    wave *= envelope

    # 加一点白噪声当"嗓音气声"
    wave += 0.02 * rng.standard_normal(len(t))

    # 归一化到 [-0.8, 0.8]，留 20% 余量防止削波
    peak = np.abs(wave).max()
    return wave / peak * 0.8


def make_sentence(sr: int, seed: int = 0) -> np.ndarray:
    """一句话 = 说话段 + 静音段 交替拼接（模拟'词语-停顿-词语'节奏）"""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(rng.integers(3, 6)):  # 随机 3~5 个"词"
        word = synth_speech_like(sr, rng.uniform(0.4, 1.2), seed=seed)
        parts.append(word)
        parts.append(0.08 * rng.standard_normal(int(sr * rng.uniform(0.2, 0.5))))  # 静音=底噪
    return np.concatenate(parts).astype(np.float32)


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    sr = 22050  # 故意用非 16kHz 的采样率，检验 D2 的重采样是否生效

    cases = [
        ("test_正常1.wav", lambda: make_sentence(sr, seed=1), 1),
        ("test_正常2.wav", lambda: make_sentence(sr, seed=2), 1),
        ("test_正常3.wav", lambda: make_sentence(sr, seed=3), 1),
        # 故意生成超长音频（拼接多句话到 40s+，应被 D2 标记"过长，待切分"）
        ("test_超长40s.wav",
         lambda: np.concatenate([make_sentence(sr, seed=40 + i)
                                 for i in range(10)]), 1),
        # 几乎全静音（应被 D2 按 SNR 剔除）
        ("test_全静音.wav",
         lambda: (0.005 * np.random.default_rng(9).standard_normal(sr * 3)).astype(np.float32), 1),
    ]

    for name, gen, _ in cases:
        wav = gen()
        path = config.DATA_RAW / name
        sf.write(path, wav, sr)  # soundfile 写 wav：传 (路径, 波形, 采样率)
        print(f"已生成 {name}（{len(wav)/sr:.1f}s @ {sr}Hz）")

    print(f"\n✅ 5 条测试音频已放入 {config.DATA_RAW}")
    print("   真实数据到位后可直接删除这些 test_* 文件。")


if __name__ == "__main__":
    main()
