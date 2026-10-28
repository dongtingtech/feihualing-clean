# -*- coding: utf-8 -*-
"""
d4b_correct_with_tangshi.py —— 用《全唐诗》标准答案自动批改机器转写
==================================================================
思路（这就是"有答案的考试"）：
1) 文件名自带"诗题 作者"（如 `F__九日遇雨二首 薛涛.wav`）——这批数据的隐藏彩蛋
2) 拿诗题+作者去 data/tangshi.jsonl（extract_tangshi.py 产出的诗库）里查标准答案
3) 机器转写 vs 标准答案做编辑距离比对（difflib.SequenceMatcher，同 D5 去重算法）：
   - 相似度 >= ACCEPT_SIM：判定"确实是这首诗"，文本直接替换为标准答案，自动纠错完成
   - 相似度 <  ACCEPT_SIM：不确定，保留机器文本、标记人工复核（宁可不改，不能改错）
4) 顺带产出数据质量指标：错字率（1-相似度）分布

比喻：机器听写像学生默写，全唐诗是课本。相似度高 = 默写基本对，直接抄课本订正；
相似度低 = 可能默串了诗（同名诗有好几首），交给老师人工看。

运行：python scripts/d4b_correct_with_tangshi.py
输出：output/transcripts_corrected.jsonl + reports/D4_全唐诗对照校对报告.md
"""
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

sys_path = Path(__file__).resolve().parent
sys_path_str = str(sys_path)
import sys
sys.path.insert(0, sys_path_str)
import config

from zhconv import convert as zh_convert

ACCEPT_SIM = 0.45      # 相似度门槛：达到才敢"直接抄答案"
MAX_CANDIDATES = 30    # 同名诗太多时，最多比对前 30 个候选

# 去掉所有标点/空白，只留汉字数字字母（比对用的"裸文本"）
PUNCT = re.compile(r"""[，。？！、；：""''《》〈〉（）()\[\]·\.\,\?\!\;\:\"'—\-…\s]""")


def norm(s: str) -> str:
    """归一化：繁转简 + 去标点空白 + 小写"""
    s = zh_convert(s or "", "zh-cn")
    s = PUNCT.sub("", s).lower()
    return s


BRACKET_TITLE = re.compile(r"[《【]([^《》【】]+)[》】]")
DECOR = re.compile(r"[《》【】（）()「」﹁﹂·。,，、\s]")


def parse_filename(name: str):
    """从音频文件名取 (诗题, 作者)；解析不出返回 (None, None)
    兼容四种格式：
      F__九日遇雨二首 薛涛            ← 题目 作者
      古诗朗诵音频合集__《渔父》张志和 ← 《题目》作者
      古诗朗诵音频合集__【寒食寄李补阙】郭郧 ← 【题目】作者
      刘笑宇__仙乐侑席_吕岩 刘笑宇    ← 题目_作者 录音人
    """
    stem = Path(name).stem
    if "__" in stem:
        stem = stem.split("__")[-1]
    stem = stem.strip()
    # A) 书名号/方括号包题目
    m = BRACKET_TITLE.search(stem)
    if m:
        title = m.group(1)
        author = DECOR.sub("", stem[m.end():])
        return (title, author) if author else (title, None)
    # B) 下划线分隔：题目_作者 [录音人]
    if "_" in stem:
        title, rest = stem.split("_", 1)
        author = rest.split()[0] if rest.split() else ""
        return title.strip(), author or None
    # C) 空格分隔：题目 作者
    if " " in stem:
        title, author = stem.rsplit(" ", 1)
        return DECOR.sub("", title).strip() or None, author.strip()
    # D) 没有分隔符：整串当诗题（作者未知，如"三月三日登庾楼，寄庾三十二"）
    title = DECOR.sub("", stem).strip()
    return (title, None) if title else (None, None)


def load_corpus():
    """读诗库，按归一化诗题建索引"""
    corpus = []
    idx = {}
    with open(config.ROOT / "data" / "tangshi.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            p["_题norm"] = norm(p["题"])
            p["_作者norm"] = norm(p["作者"])
            p["_正文norm"] = norm(p["正文"])
            corpus.append(p)
            idx.setdefault(p["_题norm"], []).append(p)
    return corpus, idx


def find_best(machine_norm: str, title_norm: str, author_norm: str, idx):
    """查诗库：先精确（题+作者），再只按诗题，取正文相似度最高的候选
    返回 (诗, 相似度, 覆盖率)。覆盖率 = 原诗被机器文本的连续匹配块盖住的比例，
    用来容忍朗读前后带的"大家好我是xxx"等额外说话内容。"""
    cands = []
    if title_norm in idx:
        exact = [p for p in idx[title_norm] if p["_作者norm"] == author_norm]
        cands = exact if exact else idx[title_norm]
    if not cands:
        return None, 0.0, 0.0
    best, best_sim, best_cov = None, 0.0, 0.0
    for p in cands[:MAX_CANDIDATES]:
        sm = SequenceMatcher(None, machine_norm, p["_正文norm"])
        sim = sm.ratio()
        cov = sum(b.size for b in sm.get_matching_blocks()) / max(len(p["_正文norm"]), 1)
        if sim > best_sim:
            best, best_sim, best_cov = p, sim, cov
    return best, best_sim, best_cov


def main():
    t0 = time.time()
    corpus, idx = load_corpus()
    print(f"[批改] 诗库载入：{len(corpus)} 首", flush=True)

    wavs = sorted((config.DATA_CLEAN).glob("*.wav"))
    machine = {}
    with open(config.OUTPUT / "transcripts.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            machine[r["音频"]] = r
    print(f"[批改] 机器转写：{len(machine)} 条", flush=True)

    corrected, unmatched, flagged = [], [], []
    sims = []
    for w in wavs:
        rel = str(w.relative_to(config.ROOT)).replace("\\", "/")
        rec = machine.get(rel)
        if rec is None:
            continue
        title, author = parse_filename(w.name)
        out = {"音频": rel, "原转写": rec["文本"], "时长s": rec["时长s"]}
        if title is None:
            out.update({"来源": "机器", "文本": rec["文本"]})
            unmatched.append(rel)
            corrected.append(out)
            continue
        t_norm, a_norm = norm(title), norm(author) if author else ""
        if not t_norm:
            unmatched.append(rel)
            out.update({"来源": "机器", "文本": rec["文本"]})
            corrected.append(out)
            continue
        poem, sim, cov = find_best(norm(rec["文本"]), t_norm, a_norm, idx)
        if poem and (sim >= ACCEPT_SIM or cov >= 0.6):
            # 相似度足够高，或原诗六成以上被连续匹配块盖住 → 确认是这首诗
            out.update({"来源": "全唐诗", "文本": poem["正文"],
                        "题": poem["题"], "作者": poem["作者"],
                        "卷序": f"卷{poem['卷']}_{poem['序']}",
                        "相似度": round(sim, 3), "覆盖率": round(cov, 3)})
            sims.append(sim)
        elif poem:
            # 找到了候选但像得不够 → 不敢自动改，标人工复核
            out.update({"来源": "人工复核", "文本": rec["文本"],
                        "疑为": f"{poem['题']}（{poem['作者']}）相似度{sim:.2f} 覆盖{cov:.2f}"})
            flagged.append(out)
        else:
            out.update({"来源": "机器", "文本": rec["文本"]})
            unmatched.append(rel)
        corrected.append(out)

    out_path = config.OUTPUT / "transcripts_corrected.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in corrected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tangshi = sum(1 for r in corrected if r["来源"] == "全唐诗")
    n_review = len(flagged)
    n_machine = sum(1 for r in corrected if r["来源"] == "机器")
    err = [1 - s for s in sims]
    mean_err = statistics.mean(err) * 100 if err else 0
    med_err = statistics.median(err) * 100 if err else 0

    # ---------- 报告 ----------
    L = ["# D4.5 全唐诗对照校对报告\n",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
         "## 一、结果概览\n",
         "| 指标 | 数值 |", "|---|---|",
         f"| 诗库 | {len(corpus)} 首（全唐诗 PDF 提取） |",
         f"| 自动纠错成功（直接采用原诗） | **{n_tangshi} 条** |",
         f"| 相似度不足、标人工复核 | {n_review} 条 |",
         f"| 文件名无诗题信息（合集等） | {len(unmatched)} 条 |",
         f"| 自动纠错部分的错字率 | 平均 {mean_err:.1f}%，中位 {med_err:.1f}% |",
         f"| 耗时 | {time.time()-t0:.0f}s |",
         "\n> 判定规则：机器文本与原诗相似度 ≥ " f"{ACCEPT_SIM}" " 才自动替换为原诗；"
         "低于门槛的宁可保留机器文本，标\"人工复核\"——宁可不改，不能改错。\n",
         "\n## 二、自动纠错样例（前 6 条，看\"听写 → 标准答案\"的差距）\n",
         "| 音频 | 机器听写（前40字） | 标准答案（前40字） | 相似度 |",
         "|---|---|---|---|"]
    shown = 0
    for r in corrected:
        if r["来源"] == "全唐诗" and shown < 6:
            L.append(f"| {Path(r['音频']).name} | {r['原转写'][:40]} | {r['文本'][:40]} | {r['相似度']} |")
            shown += 1

    if flagged:
        L += ["\n## 三、待人工复核（相似度不足，前 10 条）\n",
              "| 音频 | 机器文本（前30字） | 疑为 |", "|---|---|---|"]
        for r in flagged[:10]:
            L.append(f"| {Path(r['音频']).name} | {r['文本'][:30]} | {r['疑为']} |")

    if unmatched:
        L += [f"\n## 四、未匹配清单（{len(unmatched)} 条，多为\"卷NNN\"多诗合集录音）\n",
              "这些录音一条里连读多首诗，需要先按 D3 语段切分再逐段对诗，明日 D5 一并处理。\n"]

    L += ["\n## 五、产物\n",
          f"- `{out_path.relative_to(config.ROOT)}`：{len(corrected)} 条，"
          "新增字段 来源/相似度/卷序，下游 D5 直接用\"文本\"列",
          "- 错字率 = 1 - 编辑距离相似度，这个指标以后可以用来评估换模型值不值"]

    (config.REPORTS / "D4_全唐诗对照校对报告.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:24]), flush=True)
    print(f"[批改] 完成：自动纠错 {n_tangshi}，复核 {n_review}，未匹配 {len(unmatched)}", flush=True)


if __name__ == "__main__":
    main()
