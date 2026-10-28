# -*- coding: utf-8 -*-
"""
d5_dedupe_manifest.py —— 第 5 天·去重 + 打包 manifest + 总结
============================================================
1) 文本去重：相似度 > 0.9 的两条转写判为重复（编辑距离归一化）
   比喻：图书馆编目——每本书贴好唯一标签再上架，绝不放两本一样的。
2) 生成 manifest.jsonl：每行 = {音频路径, 文本, 时长}，ASR 训练的标准入场券
3) 汇总 D2~D4 全部报告，生成《数据清洗总结报告》

学到的技术：
- 编辑距离（Levenshtein）：两个字符串最少要几次"增/删/改"才能互相变过去
- 数据集管理的"可追溯"原则：每个样本都能查到它从哪来、为什么保留

运行：python scripts/d5_dedupe_manifest.py
输出：output/manifest.jsonl + reports/D5_总结报告.md
"""

import difflib
import json
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def similarity(a: str, b: str) -> float:
    """序列相似度 = 1 - 编辑距离/较长串长度（difflib 内置实现）"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def main():
    # 优先用全唐诗纠正后的文本（有标准答案的标签），没有则退回原始转写
    tr_corr = config.OUTPUT / "transcripts_corrected.jsonl"
    tr_path = tr_corr if tr_corr.exists() else config.OUTPUT / "transcripts.jsonl"
    print(f"[D5] 使用文本源：{tr_path.name}")
    if not tr_path.exists():
        print("output/transcripts.jsonl 不存在 —— 请先运行 d4_transcribe.py")
        sys.exit(1)

    records = [json.loads(l) for l in
               tr_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---------- ① 文本去重（贪心：按顺序两两比较） ----------
    # 判"重复"要同时满足两个条件：
    #   1) 文本相似度 > 阈值；2) 同一个录制者（文件名 "__" 前的来源前缀）
    # 只有文本像还不够——不同人读同一首诗文本几乎全同，但那是多说话人数据，
    # 恰恰是 ASR 最想要的，不能删；同一人重复上传的同一段录音才叫重复。
    def recorder(rel_path: str) -> str:
        name = Path(rel_path).name
        return name.split("__")[0] if "__" in name else name

    kept, dropped_dup = [], []
    for r in records:
        dup = None
        for k in kept:
            if recorder(r["音频"]) != recorder(k["音频"]):
                continue                      # 不同人读同一首诗 → 保留
            if similarity(r["文本"], k["文本"]) > config.SIMILARITY_THRESHOLD:
                dup = k
                break
        if dup:
            dropped_dup.append({"音频": r["音频"], "与谁重复": dup["音频"],
                                "文本": r["文本"][:40]})
        else:
            kept.append(r)

    # ---------- ② 生成 manifest.jsonl ----------
    manifest_path = config.OUTPUT / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in kept:
            wav_abs = config.ROOT / r["音频"]
            try:
                info = sf.info(str(wav_abs))
                dur = round(info.duration, 2)
            except Exception:
                dur = r.get("时长s", 0)
            f.write(json.dumps({
                "audio": r["音频"],
                "text": r["文本"],
                "duration": dur,
            }, ensure_ascii=False) + "\n")

    # ---------- ③ 汇总总结报告 ----------
    d2 = json.loads((config.REPORTS / "D2_detail.json").read_text(encoding="utf-8")) \
        if (config.REPORTS / "D2_detail.json").exists() else {}
    d3 = json.loads((config.REPORTS / "D3_detail.json").read_text(encoding="utf-8")) \
        if (config.REPORTS / "D3_detail.json").exists() else []

    total_in = len(d2.get("kept", [])) + len(d2.get("dropped", []))
    # D4 的字段名是"时长s"，manifest 里改叫 duration，两处都要认，否则总时长恒为 0
    total_dur = sum(r.get("时长s", r.get("duration", 0)) for r in kept)

    lines = [
        "# D5 数据清洗总结报告\n",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
        "## 数据漏斗（每一层都做了什么、丢了什么）\n",
        f"- 原始输入：{total_in} 条",
        f"- D2 质量过滤后：{len(d2.get('kept', []))} 条（剔除 {len(d2.get('dropped', []))} 条）",
        f"- D3 切分后：{sum(s.get('切出语段', 0) for s in d3)} 段",
        f"- D4 转写成功：{len(records)} 段",
        f"- D5 去重后保留：{len(kept)} 段（去重 {len(dropped_dup)} 段）",
        # 时长短时用分钟显示，避免出现"0.00 小时"这种看着像没数据的写法
        f"- **最终数据量：{len(kept)} 条 / 约 {total_dur/3600:.2f} 小时"
        f"（≈ {total_dur/60:.1f} 分钟，合计 {total_dur:.0f} 秒）**\n",
        "## 产物清单\n",
        f"- `output/manifest.jsonl` —— 训练用标准格式（{manifest_path}）",
        f"- `output/segments/` —— 切分降噪后的音频",
        f"- `data/clean/` —— 16kHz 统一格式的音频\n",
    ]
    if dropped_dup:
        lines.append("## 去重明细\n")
        lines += [f"- {d['音频']} ≈ {d['与谁重复']}：「{d['文本']}…」" for d in dropped_dup]

    report = config.REPORTS / "D5_总结报告.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n报告已保存：{report}")
    print(f"manifest 已生成：{manifest_path}")


if __name__ == "__main__":
    main()
