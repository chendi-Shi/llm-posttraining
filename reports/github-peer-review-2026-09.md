# GitHub 后训练项目对标与公开证据核验

2026 年 9 月 30 日核对三个公开仓库后，本项目最值得立即加强的地方是**让公开报告的数字和评测口径接受自动检查**。现有研究已记录许可、原句去重、模型与数据哈希、SFT／DPO 对照及失败门禁；公开仓库不包含原始 MASSIVE 数据、权重和逐题预测，因此读者无法仅从仓库重跑推理。新增的离线核验脚本能检查公开报告之间的内部一致性，并在 GitHub CI 中阻止不一致的更新；它不能冒充独立复现实验。

## 对标依据

下表只描述已读到的仓库文档和本项目的公开材料，不把他人的 README 数字视为独立验证过的实验结果。三个项目的数据、硬件和任务不同，分数不能横向排名。

| 项目与已核对材料 | 可借鉴做法 | 本项目现状与判断 |
| --- | --- | --- |
| [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook/blob/53c11c328684f965162f41ff93185bb9e0448263/README.md)及其[偏好方法比较](https://github.com/huggingface/alignment-handbook/blob/53c11c328684f965162f41ff93185bb9e0448263/recipes/pref_align_scan/README.md) | 用可版本化的 recipe 固定 SFT／DPO 参数，并在偏好比较中明确模型、数据和超参数 | 本项目已有脚本、运行清单、哈希和固定门槛；关键实验分散在多份报告中，公开读者仍需人工串联。优先做跨报告自动核验，避免改动已冻结的训练方案。 |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory/tree/ce9dc9e072f80fa3abe0989d4ab90da25f083438) | 将微调、评测和推理流程标准化，覆盖多方法与多模型 | 本项目只声称本机 0.5B、CPU 4-bit QLoRA 的具体实验；规模与方法覆盖不能与框架仓库直接比较。保持任务聚焦，保留精确模型与生成协议。 |
| [CodeAlign 的项目报告](https://github.com/Sergasgr/CodeAlign/blob/67561d1d3d0a94f3e31e4acda346d5578d5d5cf8/README.md) | README 同时展示对照组、评测协议、负结果及奖励设计的失败分析 | 本项目也保留同样本继续 SFT、DPO 的拒判与阳性召回权衡；缺少一个可由外部读者直接运行的公开证据校验入口。 |

[Hugging Face Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook/blob/main/contents/troubleshooting/troubleshooting-reproducibility.md)提醒，代码实现、提示模板、归一化和同名指标的定义差异都可能改变分数。本项目的自然频率开发集比较因此需要自动确认同一数据指纹、基座与 tokenizer、生成设置、参考预测链接和指标算术；这些检查比增加一张没有相同评测口径的分数表更有价值。后半句是针对本项目的判断。

## 已实施的改进

运行 `python scripts/audit_public_results.py`，仅依赖 Python 标准库和公开 JSON。脚本目前检查 v1 选择锁与确认集指纹、v1 三模型的实体计数与配对差值，以及 v2 SFT、v3 DPO、同样本 SFT 对照的 250 条开发集身份、基座、tokenizer、自由生成设置、逐题预测哈希引用、实体指标、无槽失败和配对差值。它从报告数值重新计算 v3 的八项开发集门槛与最终停止决定，并由 `.github/workflows/ci.yml` 每次提交运行。篡改 F1、参考预测哈希、生成长度或门槛决策的测试会失败。

该核验**不读取 Git 忽略的数据、权重或逐题预测，也不重新运行 bootstrap**。因此，它证明的是公开记录内部没有这些具体矛盾；训练与推理复现仍需按[复现步骤](../SLOT_V3_RUNBOOK.md)重建第三方数据与模型，并核对已发布的文件 SHA-256。当前 400 条 v2/v3 确认集继续封存，不因对标或新增审计脚本而打开。

## 对简历展示的判断

本项目适合展示应用后训练中的**数据治理、固定评测、强基线、同样本对照和负结果归因**。与通用训练框架相比，它没有多卡、7B 规模、多个随机种子或真实流量评测的证据；这些能力需要新的资源与独立设计。新增公开核验的价值在于让面试官能在不下载模型的情况下检查核心报告是否自洽，同时清楚看到哪些结论仍依赖未公开的原始预测。
