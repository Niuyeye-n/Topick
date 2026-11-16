# Topick

TopicK 在 CoopES 论文实验设置中的适配代码与实验记录。不是作者官方仓库，也不是 CoopES 原实现。

本实验只训练约 177 万参数的主题预测器，冻结 all-mpnet-base-v2 编码器和两个 Reader。离线主题标注使用本地 Llama-3-8B-Instruct，未调用付费 API。每个任务使用 300 道官方训练题，内部 270/30 划分。候选池 50、4-shot、dev 50、test 500。

**OpenBookQA 使用 additional 配置：所有示例及目标均提供 gold fact1（oracle 条件）**。该成绩不能当作不提供事实的 OpenBookQA 成绩。Qwen 是 Qwen2.5-7B Base，Llama 是 Llama-3-8B-Instruct；实际制品身份由 `configs/reader_reference.json` 中的文件哈希约束。

## 已有结果

单种子 seed=42，全部为百分数。run02 是统一采用的最终轮，run01 是保留的学习不足诊断轮。

| 数据集 | Reader | run02 λ | 正确/总数 | Acc | Weighted-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| OpenBookQA + fact1 | Llama-3-8B-Instruct | 0.1 | 449/500 | 89.80 | 89.8020 | 89.6337 |
| OpenBookQA + fact1 | Qwen2.5-7B Base | 1.0 | 459/500 | 91.80 | 91.9177 | 91.6645 |
| ARC-C | Llama-3-8B-Instruct | 0.5 | 412/500 | 82.40 | 82.3859 | 82.4368 |
| ARC-C | Qwen2.5-7B Base | 0.5 | 440/500 | 88.00 | 87.9937 | 88.0882 |

λ 是 `Z(topic) + λ Z(semantic)` 的语义权重。仅在 dev50 上从 0.1/0.5/1.0 选取，平局取小值。没有在未选中 λ 上运行 test。`results/metrics_summary.json` 保留两轮全部指标。

## 文件

- `snapshots/run01/code/`、`snapshots/run02/code/`：实际执行源码的精确快照；哈希见 `configs/source_manifest.json`。原作者对照脚本未重新分发。
- `code/`：便于异地运行的入口。仅改路径配置、资源下载和新运行准备编排；训练损失、选择算法、run02 训练及受约束字母评测保留。与快照的差异可直接 diff。
- `configs/frozen_ids.json`：历史候选池、dev、test 以及主题监督划分的官方 ID，不包含数据集文本或权重。
- `scripts/prepare_inputs.py`：从公开官方数据重建冻结输入，并强制核对历史哈希。
- `scripts/verify_results.py`：从单独交付的完整结果包核对文件和 8 组逐题指标，无需 GPU。
- `results/`：两轮指标、token 统计、报告和训练曲线；完整逐题结果通过单独 ZIP 交付。

## 安装与运行

执行环境是 Linux / Python 3.12.16 / 单张 RTX 4090 / Torch 2.8.0 + CUDA 12.8。依赖版本见 `requirements.txt` 和 `configs/environment.json`；这里没有声称已在另一台全新机器完成端到端复跑。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 使用新目录，不能指向已完成实验目录。
python scripts/prepare_inputs.py --run-dir /data/topick/runs/run03
export TOPICK_RUN_DIR=/data/topick/runs/run03
export TOPICK_ASSETS_DIR=/data/topick/assets
export TOPICK_QWEN_MODEL=/data/models/qwen2.5-7b
export TOPICK_LLAMA_MODEL=/data/models/llama3-8b-instruct
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
python "$TOPICK_RUN_DIR/code/pipeline.py"
```

输入准备会下载官方 ARC-C / OpenBookQA 数据并写入新运行目录；首次流水线下载固定revision（e8c3b32edf5434bc2275fc9bab85f82640a19130）的编码器并核对历史权重SHA-256，以及固定版 CMS 主题表，进行本地标注、软标签构造、run02 训练、Reader 知识估计、dev 选 λ 和 test。Reader 必须自行取得，代码会在加载前检查历史文件哈希，哈希不符会停止。更换制品需单独冻结新协议，不能静默去掉校验。

run02 训练是 AdamW、batch64、学习率 1e-5/1e-4、上限1000轮、最少50轮、验证损失连续100轮未改善早停。历史 run02 复用 run01 标注与向量，因此约5分钟的流水线耗时不代表从零准备耗时。历史 run01 使用30轮，源码保留但不是默认入口。新标注可能受软件/硬件影响而不逐位相同；未附历史训练语料或权重，不能承诺从零重建精确同分。

离线检查结果 ZIP 解压目录：

```bash
python scripts/verify_results.py /data/TopicK_complete_results
```

## 对照与限制

本地 CoopES-R 在同一冻结 Reader、数据、prompt 和打分协议下另有结果；其全局贡献 dev50、均匀权重、M=50 是重实现约定，不能称为原作者实现。原稿 Table I 是外部参考，原测试 ID、精确模型制品及实际运行配置未完全取得。单次探索性测试和缩小的训练预算不能证明相对原作者方法的普遍优势。报告给出同协议 CoopES-R 与原稿外部参考两类对照。

代码/结果不包含连接凭据、原始训练语料或大模型权重。第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
