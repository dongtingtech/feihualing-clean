# -*- coding: utf-8 -*-
"""
d4_transcribe.py —— 第 4 天·自动转写 + 文本清洗
================================================
任务：
1) 用 Whisper 把 data/clean/*.wav（D2 归一后的整条录音）批量转成文字
2) 文本清洗：繁转简、口头语处理、标点/空白规整、幻觉句过滤

为什么要"整条转写"而不是"逐段转写"（2026-10-27 实测，见 reports/D4_转写速度与策略实测.md）：
Whisper 的编码器固定吃 30 秒窗口，输入 0.46 秒也要补到 30 秒算一遍。
  A 逐段（26369 段碎片）：RTF 2.37，全量 8 小时，错字多
  B 整条（1282 条完整录音）：RTF 0.04，全量约 18 分钟，整首可读
比喻：叫一辆 30 座大巴，A 每趟只送 1 个人，B 每趟坐满 —— 固定成本摊薄 59 倍，
和分批梯度下降里 batch size 太小的道理一样。

口头语的两种策略（这是规范决策，最终由你定）：
  A. delete    —— 删除，数据更"干净"，适合训练 TTS/清晰朗读模型
  B. normalize —— 保留但转规范书写，保留口语真实分布，适合训练口语 ASR
本脚本默认 B，改下面 FILLER_POLICY 即可切换。

运行：
    python scripts/d4_transcribe.py            # 全量
    python scripts/d4_transcribe.py --limit 50 # 只跑前 50 条（估时用）
输出：
    output/transcripts.jsonl                 每行 = 音频路径 + 文本 + 时长
    output/今日校对/10条_待校对.md            给你人工批改的样本 + 3 条标注规则草案
    reports/D4_转写报告.md
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ====== 规范开关（你要定的"3 条标注规则"就改这里） ======
FILLER_POLICY = "normalize"   # "delete"=删掉嗯啊 / "normalize"=转书面并保留
PUNCTUATION = True            # 是否保留标点（Whisper 自带标点能力）
VAD_FILTER = False            # D3 已做过 VAD，这里不再重复（重复会丢掉轻声有效短句）
BEAM_SIZE = 1                 # 1=贪心解码，快 3~4 倍；嫌错字多可改 5（更准更慢）

# 口头语映射表：口语 → 书面
FILLER_MAP = {"嗯": "", "呃": "", "啊": "", "那个那个": "那个", "然后然后": "然后"}

# Whisper 在静音/噪声上的经典"幻觉句"：模型没听到内容时会编出这几句
HALLUCINATION = (
    "字幕由Amara.org社区提供", "字幕提供者", "字幕由", "请不吝点赞",
    "订阅", "谢谢观看", "谢谢大家", "明镜与点点栏目", "MING PAO CANADA",
)

SAMPLE_FOR_HUMAN = 10         # 抽几条给你人工校对


def to_simplified(text: str) -> str:
    """繁体 → 简体（base 模型默认吐繁体：萬里京標 → 万里京标）"""
    try:
        from zhconv import convert
        return convert(text, "zh-cn")
    except ImportError:
        return text            # 没装就原样返回，不阻塞


def clean_text(text: str) -> str:
    """文本清洗：口径统一，但不动语义"""
    t = text.strip()
    # 1) 幻觉句直接整条丢弃
    for h in HALLUCINATION:
        if h and h in t:
            return ""
    # 2) 口头语
    if FILLER_POLICY == "delete":
        for k in FILLER_MAP:
            t = t.replace(k, "")
    else:  # normalize：只压掉"嗯嗯嗯"这种重复，不删词
        for k, v in FILLER_MAP.items():
            if v and len(k) == 1:
                t = re.sub(rf"({re.escape(k)}){{2,}}", v, t)
            elif v:
                t = re.sub(rf"({re.escape(k)})+", v, t)
    # 3) 繁转简
    t = to_simplified(t)
    # 4) 空白与标点规整
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([，。？！、；：,.?!])", r"\1", t)
    # 半角标点统一成全角（Whisper 中文输出常混半角逗号句号，口径要统一）
    t = t.translate(str.maketrans({",": "，", "?": "？", "!": "！",
                                   ":": "：", ";": "；", ".": "。"}))
    t = re.sub(r"([，。？！、；：])\1+", r"\1", t)      # 重复标点压成一个
    if PUNCTUATION and t and t[-1] not in "。？！.?!…":
        t += "。"                                     # 句末补句号，口径统一
    return t.strip()


def load_model():
    from faster_whisper import WhisperModel
    local_model = config.TOOLS / "faster-whisper-base"
    model_name = str(local_model) if (local_model / "model.bin").exists() else "base"
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    args = ap.parse_args()

    src_dir = config.DATA_CLEAN
    wavs = sorted(src_dir.glob("*.wav"))
    if not wavs:
        print("data/clean 为空 —— 请先运行 d2_format_and_filter.py")
        sys.exit(1)
    if args.limit:
        wavs = wavs[: args.limit]

    out_path = config.OUTPUT / "transcripts.jsonl"
    tag = f"抽样{len(wavs)}条" if args.limit else "全量"
    print(f"[D4] 待转写 {len(wavs)} 条（{tag}），模型 base / int8 / beam={BEAM_SIZE}")

    try:
        model = load_model()
    except ImportError:
        msg = ("⚠️ 未安装 faster-whisper，转写任务【留存】未执行。\n"
               "   安装：pip install faster-whisper")
        print(msg)
        (config.REPORTS / "D4_转写报告.md").write_text(msg, encoding="utf-8")
        sys.exit(0)

    import soundfile as sf

    # 断点续跑：已完成的行直接跳过（每转完一条立刻落盘，中断也不丢成果）
    done_keys, results = set(), []
    if out_path.exists() and not args.limit:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # 中断时可能写了半行，丢掉
            done_keys.add(r["音频"])
            results.append(r)
        print(f"[D4] 检测到已有 {len(done_keys)} 条，续跑剩余部分")

    errors, empties = [], []
    zh_hit = 0
    t0 = time.time()
    audio_sec = sum(r.get("时长s", 0) for r in results)
    todo = [w for w in wavs if str(w.relative_to(config.ROOT)).replace("\\", "/") not in done_keys]
    print(f"[D4] 本次待转写 {len(todo)} 条", flush=True)

    with open(out_path, "a" if done_keys else "w", encoding="utf-8") as fout:
        for i, w in enumerate(todo, 1):
            try:
                # soundfile 解码成 float32 数组直接喂模型（绕开内部 av 解码器的兼容问题）
                audio, sr = sf.read(str(w), dtype="float32")
                segments, info = model.transcribe(audio, language="zh",
                                                  beam_size=BEAM_SIZE,
                                                  vad_filter=VAD_FILTER,
                                                  condition_on_previous_text=False)
                raw = "".join(s.text for s in segments).strip()
                if not raw:
                    empties.append(w.name)
                    continue
                text = clean_text(raw)
                if text != raw:
                    zh_hit += 1
                if not text:
                    empties.append(w.name)
                    continue
                dur = float(info.duration)
                audio_sec += dur
                rec = {
                    "音频": str(w.relative_to(config.ROOT)).replace("\\", "/"),
                    "文本": text,
                    "时长s": round(dur, 2),
                    "字数": len(text),
                }
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()          # 立刻落盘：中断最多损失 1 条
            except Exception as e:
                errors.append({"音频": w.name, "错误": f"{type(e).__name__}: {e}"})

            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}  已用 {el/60:.1f} 分  预计剩余 {eta/60:.1f} 分", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["音频"])

    # ---------- 统计 ----------
    total_chars = sum(r["字数"] for r in results)
    rtf = audio_sec / elapsed if elapsed else 0
    avg_chars = total_chars / len(results) if results else 0
    empties = [e for e in empties if e]

    # ---------- 报告 ----------
    L = ["# D4 转写报告（Whisper base · 整条转写）\n",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
         "## 一、结果概览\n",
         "| 指标 | 数值 |",
         "|---|---|",
         f"| 输入音频 | {len(wavs)} 条（`data/clean/*.wav`，{tag}） |",
         f"| 成功转写 | {len(results)} 条 |",
         f"| 空文本/幻觉句被丢 | {len(empties)} 条 |",
         f"| 失败 | {len(errors)} 条 |",
         f"| 音频总长 | {audio_sec/60:.1f} 分钟 |",
         f"| 总耗时 | {elapsed/60:.1f} 分钟 |",
         f"| 实时率 RTF | {rtf:.2f}（处理速度是播放速度的 {rtf:.1f} 倍） |",
         f"| 总字数 | {total_chars} 字，平均 {avg_chars:.0f} 字/条 |",
         f"| 文本经过清洗改动 | {zh_hit} 条（繁转简/口头语/标点） |",
         "\n## 二、文本清洗口径（这 3 条就是我替你起草的标注规则草案）\n",
         f"1. **简繁**：一律转简体（`zhconv` 繁转简，{zh_hit} 条命中）",
         f"2. **口头语**：当前策略 `{FILLER_POLICY}`（normalize=保留但转书面；delete=直接删）",
         f"3. **标点**：{'保留 Whisper 标点，句末补齐句号' if PUNCTUATION else '全部去标点'}；"
         "重复标点压成一个，标点前不留空格",
         "\n> 另加一条保险：静音/噪声上 Whisper 会编出\"字幕由…提供\"\"谢谢观看\"之类的幻觉句，"
         f"这类整条丢弃（本次丢 {len(empties)} 条）。\n",
         "\n## 三、抽样检查（前 8 条，看机器标得怎么样）\n",
         "| 音频 | 机器文本 |", "|---|---|"]
    for r in results[:8]:
        L.append(f"| {Path(r['音频']).name} | {r['文本'][:60]} |")

    L += ["\n## 四、为什么整条转写比逐段快 59 倍\n",
          "Whisper 编码器固定吃 30 秒窗口——输入 0.46 秒也补到 30 秒算一遍。",
          "总成本 = 调用次数 × 单次固定开销 + 音频长度 × 单位成本。碎片化把第一项放大到失控。",
          "这就是**固定成本摊销**：和 batch size 太小导致训练慢是一回事。",
          "代价是失去了逐段边界，所以文本按「整条录音」对齐；如要字级时间戳再往下切，"
          "D3 的能量 VAD 已给出语段边界，两者可按时间轴合并。\n",
          f"\n## 五、产物\n",
          f"- `{out_path.relative_to(config.ROOT)}`：{len(results)} 行，每行 {{音频, 文本, 时长s, 字数}}",
          "- `output/今日校对/10条_待校对.md`：给你人工批改的 10 条样本"]

    if errors:
        L.append("\n## 六、失败明细\n")
        L += [f"- {e['音频']}：{e['错误']}" for e in errors[:20]]

    (config.REPORTS / "D4_转写报告.md").write_text("\n".join(L), encoding="utf-8")

    # ---------- 人工校对样本 ----------
    sample_dir = config.OUTPUT / "今日校对"
    sample_dir.mkdir(parents=True, exist_ok=True)
    picks = results[:SAMPLE_FOR_HUMAN] if len(results) >= SAMPLE_FOR_HUMAN else results
    S = ["# D4 人工校对样本（10 条）\n",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n",
         "**怎么用**：左边点开音频听（`data/clean/` 里同名 wav），右边看机器写的字。\n",
         "只需要判断三件事：①有没有错字 ②口头语留还是删 ③标点要不要。\n",
         "\n| # | 音频文件 | 机器文本 | 你听到的（手写/打勾即可） |",
         "|---|---|---|---|"]
    for i, r in enumerate(picks, 1):
        S.append(f"| {i} | {Path(r['音频']).name} | {r['文本']} |  |")
    S += ["\n## 待你拍板的 3 条标注规则\n",
          "| 规则 | 我的建议 | 你的决定 |",
          "|---|---|---|",
          "| 口头语（嗯/啊/那个） | 保留并转书面（normalize）——口语数据更真实 |  |",
          "| 错字（古诗生僻字） | 人工改，机器只做预标注 |  |",
          "| 标点 | 保留全角标点，句末补句号 |  |"]
    (sample_dir / "10条_待校对.md").write_text("\n".join(S), encoding="utf-8")

    print("\n".join(L[:20]))
    print(f"\n[D4] 完成：{len(results)} 条，耗时 {elapsed/60:.1f} 分，RTF {rtf:.2f}")


if __name__ == "__main__":
    main()
