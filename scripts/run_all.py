# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑通 D2→D5 全流水线
======================================
比喻：流水线的"总闸"。逐个执行 4 个阶段的脚本，
任何一个环节崩了立即停下并告诉你卡在哪，不带着坏数据往下走。

用法：
    python scripts/run_all.py            # 跑全部（D2→D5）
    python scripts/run_all.py --upto d3  # 只跑到 D3（比如想先人工听听切分效果）

留存机制：D4 转写若缺 faster-whisper 会自动留存并继续（不阻塞 D5 之外环节）。
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ("D2 格式归一 + 质量过滤", "d2"),
    ("D3 VAD切分 + 降噪 + 响度归一", "d3"),
    ("D4 Whisper 转写 + 文本清洗", "d4"),
    ("D5 去重 + manifest + 总结", "d5"),
]


def main():
    upto = None
    if "--upto" in sys.argv:
        upto = sys.argv[sys.argv.index("--upto") + 1].lower()  # 如 "d3"

    for name, prefix in STAGES:
        print(f"\n===== {name} =====")
        script = HERE / f"{prefix}_*.py"
        matches = list(HERE.glob(f"{prefix}_*.py"))
        if not matches:
            print(f"❌ 找不到脚本 {script}，流水线停止")
            sys.exit(1)
        r = subprocess.run([PY, str(matches[0])], cwd=str(HERE.parent))
        if r.returncode != 0:
            print(f"\n❌ {name} 失败（exit {r.returncode}），流水线在此停止。")
            print("   修复后重新运行即可，已完成环节的结果都保留着。")
            sys.exit(r.returncode)
        if upto and prefix == upto:
            print(f"\n✅ 已按 --upto {upto} 停止。")
            return

    print("\n🎉 全流水线跑完！去 reports/ 看各环节报告，output/manifest.jsonl 是最终产物。")


if __name__ == "__main__":
    main()
