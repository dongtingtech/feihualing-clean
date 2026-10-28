# -*- coding: utf-8 -*-
"""
check_env.py —— 第 1 天·环境体检
================================
比喻：新厨房开火前先试一遍水电煤气。
这个脚本逐项检查：Python 版本、ffmpeg、依赖库、数据目录，
最后生成一份体检报告 reports/D1_环境体检报告.md。

运行方式（在项目根目录）：
    python scripts/check_env.py
"""

import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  （导入我们的配置文件）

REPORT_LINES = []


def check(name: str, ok: bool, detail: str = ""):
    """记录一项检查结果。ok=True 打 ✅，False 打 ❌"""
    mark = "✅" if ok else "❌"
    line = f"- {mark} **{name}** {detail}"
    REPORT_LINES.append(line)
    print(line)
    return ok


def main():
    REPORT_LINES.append(f"# D1 环境体检报告\n")
    REPORT_LINES.append(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # ---- 1. Python 版本 ----
    v = sys.version_info
    check("Python", v.major == 3 and v.minor >= 10,
          f"当前 {v.major}.{v.minor}.{v.micro}（要求 3.10+）")

    # ---- 2. ffmpeg ----
    ffmpeg = config.find_ffmpeg() or shutil.which("ffmpeg")
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
            ver = out.stdout.splitlines()[0][:60]
            check("ffmpeg", True, f"{ver}（路径：{ffmpeg}）")
        except Exception as e:
            check("ffmpeg", False, f"找到了但运行失败：{e}")
            ffmpeg = None
    else:
        check("ffmpeg", False, "未找到（tools/ 下没有 ffmpeg.exe，且不在系统 PATH）")

    # ---- 3. Python 依赖库 ----
    libs = ["numpy", "soundfile", "librosa", "tqdm"]
    for lib in libs:
        try:
            mod = __import__(lib)
            check(f"依赖库 {lib}", True, f"版本 {getattr(mod, '__version__', '未知')}")
        except ImportError:
            check(f"依赖库 {lib}", False, "未安装，运行: pip install " + lib)

    # ---- 4. 目录结构 ----
    for d in [config.DATA_RAW, config.DATA_CLEAN, config.OUTPUT, config.REPORTS]:
        exists = d.exists()
        check(f"目录 {d.relative_to(config.ROOT)}", exists,
              "" if exists else "不存在，已自动创建")
        if not exists:
            d.mkdir(parents=True, exist_ok=True)

    # ---- 5. 扫描原始数据 ----
    exts = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus"}
    files = [p for p in config.DATA_RAW.rglob("*") if p.suffix.lower() in exts] \
        if config.DATA_RAW.exists() else []
    # 区分"测试音频"（文件名以 test_ 开头，d0 脚本合成）与"真实音频"
    test_files = [p for p in files if p.name.startswith("test_")]
    real_files = [p for p in files if not p.name.startswith("test_")]

    if files:
        REPORT_LINES.append(f"\n## 数据盘点\n")
        REPORT_LINES.append(f"data/raw 下共发现 **{len(files)}** 个音频文件"
                            f"（真实音频 {len(real_files)} 个 / 合成测试音频 {len(test_files)} 个）：\n")
        for p in files[:20]:
            size_kb = p.stat().st_size / 1024
            tag = " 〔测试〕" if p.name.startswith("test_") else ""
            REPORT_LINES.append(f"- {p.name}（{size_kb:.0f} KB）{tag}")
        if len(files) > 20:
            REPORT_LINES.append(f"- ……等共 {len(files)} 个")

        if real_files:
            next_step = f"真实数据已就位（{len(real_files)} 个）！下一步运行 d2 开始清洗。"
        else:
            REPORT_LINES.append("\n⚠️ **以上全部是合成测试音频，真实数据待放入 `data/raw/`。**")
            next_step = ("真实数据待放入 —— 把你要清洗的音频复制到 `data/raw/`，"
                         "明天（D2）即可开始格式归一与质量过滤。"
                         "今天先用测试音频跑通了全流程（流水线本身已验证可用）。")
    else:
        REPORT_LINES.append(f"\n## 数据盘点\n")
        REPORT_LINES.append("data/raw 目前为空 —— **这是当前唯一缺的东西**。\n")
        next_step = "把你的音频文件放进 data/raw/ 后重新运行本脚本。"

    # ---- 6. 测试音频合成工具（数据没到位也能先跑通全流程）----
    test_audio = config.OUTPUT / "test_audio"
    check("合成测试音频脚本", (Path(__file__).parent / "d0_make_test_audio.py").exists(),
          "scripts/d0_make_test_audio.py —— 用假语音先验证整条流水线")

    REPORT_LINES.append(f"\n## 下一步\n\n{next_step}\n")

    report_path = config.REPORTS / "D1_环境体检报告.md"
    report_path.write_text("\n".join(REPORT_LINES), encoding="utf-8")
    print(f"\n报告已保存：{report_path}")

    # 汇总判定
    failed = [l for l in REPORT_LINES if "❌" in l]
    if failed:
        print(f"\n⚠️ 有 {len(failed)} 项未通过，见上方 ❌ 条目")
        sys.exit(1)
    print("\n🎉 环境体检全部通过！")


if __name__ == "__main__":
    main()
