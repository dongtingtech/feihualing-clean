# audio_pipeline —— 音频数据清洗流水线（五天项目）

> 用途：把任意格式的原始音频清洗成可直接喂给 ASR 训练/识别工具的标准数据集
> （16kHz 单声道 WAV + 转写文本 + manifest.jsonl）。

## 现在的状态（2026-10-28 13:20，D4 真实数据已跑完）

| 环节 | 状态 |
|---|---|
| D1 环境搭建 + 概念讲义 | ✅ 完成（ffmpeg/依赖/代码全部就绪，体检报告见 reports/） |
| **真实音频数据** | ✅ **1282 个（32 个来源，541MB，424 分钟），已放入 `data/raw/`** |
| D2 格式归一 + 质量过滤 | ✅ **真实数据已跑完：1282/1282 全通过，0 剔除，76 条超 30s 待 D3 切分** |
| D3 VAD切分 + 降噪 + 响度归一 | ✅ **真实数据已跑完：1282 条 → 26369 段，语音保留率 49.8%，仅 4 条 0 段（极轻声边缘录音）** |
| D4 Whisper 转写 + 文本清洗 | ✅ **真实数据已跑完：1282/1282 条，13.7 分钟（RTF 31），69416 字，0 失败 0 空文本** |
| D4.5 全唐诗对照自动纠错 | ✅ **688 条自动纠错（54%）+ 239 条待人工复核 + 355 条无诗题信息** |
| D5 去重 + manifest + 总结 | ✅ **已完成：1282 条全部保留（0 重复），manifest.jsonl 生成，最终 7.07 小时** |

> **D3 关键修复（2026-10-27）**：
> ① VAD 原固定阈值 0.02 对轻声朗读太高，98 个文件被误判全静音切出 0 段 → 改为自适应阈值（噪声底 = 能量 20% 分位 × 8，保底 0.004），0 段文件 98 → 4。
> ② 相邻语段因前后 100ms 缓冲互相重叠 90ms（同一音频被算两次）→ 重叠区中点对半分。
> 听感反馈与噪音量化核查：`reports/D3_听感反馈与噪音核查.md`；疑似噪音段清单：`reports/D3_疑似噪音段.json`。
> 旧版切分产物保留在 `output/_old_segments_20261026/` 和 `output/_old_segments_20261027a/`（共约 630MB，确认新版没问题后可手动删除）。

> **D4 关键决策（2026-10-28）**：
> ① 转写对象从"26369 个碎片语段"改成"1282 条完整录音"——Whisper 编码器固定吃 30s 窗口，碎片化让固定成本放大 59 倍（逐段 RTF 2.37 / 整条 RTF 0.04）。实测全量 13.7 分钟跑完，失败 0。
> ② 装 `zhconv` 做繁转简（base 模型默认吐繁体），标点半角统一全角，并加"字幕由…提供"类幻觉句过滤。
> ③ 脚本改成**逐条落盘 + 断点续跑**：中断后重跑只补未完成的部分（第一次运行时被中断，白跑了一轮）。
> ④ 已知局限：base 模型对古诗生僻字错字明显（"江娥啼竹"→"江城销锁"），定位是"机器预标注 + 人工批改"；要更准可换 small 模型（约 40 分钟）。
> 待你拍板的 3 条标注规则：`output/今日校对/10条_待校对.md`

> **D4.5 全唐诗对照自动纠错（2026-10-28，用户提供 PDF）**：
> ① `scripts/extract_tangshi.py`：从微信缓存的《全唐诗（简体电子书）(2).pdf》（15059 页）提取 40025 首诗 → `data/tangshi.jsonl`。三个坑：门牌号 `卷N_M【` 与【之间有空格；页脚"唐诗宋词全集·全唐诗"粘行；长诗题折行导致括号跨行——都要在解析时处理。
> ② `scripts/d4b_correct_with_tangshi.py`：文件名自带"诗题 作者"（4 种格式都兼容），查诗库拿标准答案，difflib 编辑距离比对：相似度 ≥0.45 或原诗 60% 被连续匹配块覆盖 → 直接替换为原诗；否则标"人工复核"（宁可不改，不能改错）。
> ③ 结果：688 条自动纠错（纠错部分机器听写与原诗相似度中位 0.55，即 base 模型近一半字是错的）、239 条待人工复核、355 条文件名无诗题信息（如"10月25日 101"，需内容反查，D5 处理）。
> 产物：`output/transcripts_corrected.jsonl`（下游 D5 优先用它的"文本"列）、`reports/D4_全唐诗对照校对报告.md`。

> 数据来源：微信「飞花令音频采集」群 → 微信本地缓存 `D:\微信yyc\xwechat_files\...\msg\file\` → 解压到 `D:\AI-poem\音频收集 第七周\` → 去重改名同步到 `data/raw/`。
> 文件名规则：`来源__原文件名`（前缀保证全局唯一，否则 D2 输出会同名覆盖）。
> 环境备忘：pip 用阿里云镜像；Whisper 模型已本地化，无需联网。
> 今日全链路自检细节与修掉的 4 个坑：`reports/D1_全链路自检报告.md`
> 数据来源分布：`reports/D2_raw数据总览.md`

## 三步上手

1. **放数据**：把你的音频文件复制到 `data/raw/`
2. **跑流水线**：`python scripts/run_all.py`
3. **看结果**：`reports/` 下每个环节一份报告，`output/manifest.jsonl` 是最终产物

## 队友首次使用（克隆后必看）

**本仓库只含代码与报告，不含音频数据和模型文件**（见 `.gitignore`，总计约 1.1MB）。

```bash
# 1) 克隆
git clone git@github.com:dongtingtech/feihualing-clean.git
cd feihualing-clean

# 2) 装依赖（建议先建虚拟环境）
pip install faster-whisper soundfile numpy librosa noisereduce pyloudnorm zhconv pymupdf -i https://mirrors.aliyun.com/pypi/simple/

# 3) 准备 ffmpeg：放一个 ffmpeg.exe 到 tools/ 目录（或装系统版并加入 PATH）
#    注意 libsndfile 不支持 m4a/aac，D2 必须用 ffmpeg 解码

# 4) 准备 Whisper 模型（不联网也能跑）：把 faster-whisper 的 base 模型放到 tools/faster-whisper-base/
#    下载：HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Systran/faster-whisper-base --local-dir tools/faster-whisper-base

# 5) 自检 → 放数据 → 跑全流程
python scripts/check_env.py
python scripts/run_all.py
```

**目录说明**：`data/raw/`（原始音频）、`data/clean/`（归一后 16kHz wav）、`data/tangshi.jsonl`（全唐诗诗库，需先跑 `scripts/extract_tangshi.py`）、`output/segments/`（切分降噪后语段）、`output/manifest.jsonl`（最终训练用清单）。

## 先看这两篇

- `docs/00_五天任务总览_按课表排布.md` —— 五天怎么排、每天你做什么
- `docs/01_概念先修课_高数线代版.md` —— 所有原理，结合高数线代讲
- `docs/02_D1今晚实操指南.md` —— 今晚 20 分钟该做的事
