# 代码与方法来源

本仓库是 TopicK 的 ARC-C/OpenBookQA 本地适配实验代码，不是作者官方仓库，也不是 CoopES 原实现。

- 论文：https://aclanthology.org/2025.emnlp-main.1007/
- 作者代码：https://github.com/WonbinKweon/TopicK_EMNLP2025
- 固定对照提交：0d5d0ba727d7cd072f8dc328ceaaee9ee949d440。
- 主题词表从固定提交的 data/cms/topic_list 获取；不在本仓库重新分发作者代码快照、词表、模型或数据集。
- code/topic_core.py 是本实验对论文目标和作者选择路径的适配实现；保留 raw-logit query ranking、0.53 rank weight、0.1 filtering、pooled selected embeddings 等行为。不能将本仓库称为作者原结果的精确复现。
- 访问模型与数据需遵守各自使用条款，Llama 权重需自行取得。没有替第三方材料授予新许可证。本仓库未额外指定开源许可证。
