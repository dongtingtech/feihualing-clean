# -*- coding: utf-8 -*-
"""
extract_tangshi.py —— 把《全唐诗》PDF 解析成结构化诗库
================================================
输入：微信文件缓存里的《全唐诗（简体电子书）(2).pdf》（15059 页，纯文本版）
输出：data/tangshi.jsonl，每行 = {"卷":N, "序":M, "题":..., "作者":..., "正文":...}

为什么做这一步：转写出来的文字是"机器的听写"，而《全唐诗》是这批古诗朗诵的
标准答案。有了答案库，就能自动对照批改（D5 的编辑距离正好派上用场）。

PDF 里每首诗的格式固定：
    圣碟电子图书            ← 页眉，丢
    ·1964·                 ← 页码，丢
    卷141_28【失题】        ← 诗的"门牌号"
    （空行）
    王昌龄                 ← 作者
    （空行）
    奸雄乃得志，遂使群心摇。... ← 正文，直到下一个"卷N_M【"为止
"""
import json
import re
import sys
import time
from pathlib import Path

import pymupdf

PDF = Path(r"D:/微信yyc/xwechat_files/wxid_mruzlsha6cfd22_7327/msg/file/2026-10/全唐诗（简体电子书）(2).pdf")
OUT = Path(__file__).resolve().parent.parent / "data" / "tangshi.jsonl"

# 页眉页脚：书名行、·数字· 页码
JUNK = re.compile(r"^(圣碟电子图书|·\d+·)$")
# 行内杂质：页脚"唐诗宋词全集·全唐诗"、"第六百六十一卷"这类卷分隔
INLINE_JUNK = re.compile(r"唐诗宋词全集·全唐诗|第[零〇一二三四五六七八九十百千]+卷")
# 诗的门牌号：卷141_28【失题】（编号与【之间可能有空格；可能出现在行内任何位置）
POEM_HEAD = re.compile(r"卷(\d+)_(\d+)\s*【([^】]*)】")


def main():
    t0 = time.time()
    doc = pymupdf.open(str(PDF))
    print(f"[诗库] 打开 PDF：{len(doc)} 页", flush=True)

    # 1) 逐页取文本，丢页眉页脚；门牌号可能粘在行中间、诗题也可能折行
    segments = []          # 元素：("head", 卷, 序, 题) 或 ("text", 字符串)
    carry = ""             # 门牌号未闭合（诗题折行）时暂存，与下一行拼上
    opener = re.compile(r"卷\d+_\d+\s*【")
    for i, page in enumerate(doc):
        for raw in page.get_text().splitlines():
            line = (carry + INLINE_JUNK.sub("\n", raw)).strip() if carry \
                else INLINE_JUNK.sub("\n", raw).strip()
            if not line or JUNK.match(line):
                continue
            carry = ""
            m = opener.search(line)
            if m and "】" not in line[m.end():]:
                # 诗题折行：先把门牌号之前的部分交出去，剩下的等下一行拼
                if m.start() > 0 and line[:m.start()].strip():
                    segments.append(("text", line[:m.start()].strip()))
                carry = line[m.start():]
                continue
            parts = []                    # 本行拆成 (是否门牌号, 内容) 序列
            pos = 0
            for mm in POEM_HEAD.finditer(line):
                if line[pos:mm.start()].strip():
                    parts.append((False, line[pos:mm.start()]))
                parts.append((True, mm))  # 门牌号
                pos = mm.end()
            if line[pos:].strip():
                parts.append((False, line[pos:]))
            for is_head, seg in parts:
                if is_head:
                    segments.append(("head", seg.group(1), seg.group(2), seg.group(3)))
                else:
                    segments.append(("text", seg.strip()))
        if (i + 1) % 2000 == 0:
            print(f"  已读 {i+1}/{len(doc)} 页，{time.time()-t0:.0f}s", flush=True)
    print(f"[诗库] 文本读取完毕，共 {len(segments)} 个片段，{time.time()-t0:.0f}s", flush=True)

    # 2) 组装诗：门牌号开新诗，第一个 text 片段若是短行（≤6字）当作者，其余进正文
    poems, cur = [], None

    def close():
        nonlocal cur
        if cur and cur["正文行"] and cur["作者"]:
            body = re.sub(r"\s+", "", "".join(cur["正文行"]))
            if body:
                poems.append({"卷": cur["卷"], "序": cur["序"], "题": cur["题"],
                              "作者": cur["作者"], "正文": body})
        cur = None

    for seg in segments:
        if seg[0] == "head":
            close()
            cur = {"卷": int(seg[1]), "序": int(seg[2]), "题": seg[3].strip(),
                   "作者": "", "正文行": []}
        elif cur is not None:
            text = seg[1].strip()
            if not cur["作者"] and 0 < len(text) <= 6 and "，" not in text and "。" not in text:
                cur["作者"] = text       # 作者行：标题后的短行
            else:
                cur["正文行"].append(text)
    close()

    # 3) 输出
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for p in poems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n += 1

    print(f"[诗库] 完成：{n} 首诗 → {OUT}，总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
