# 后训练项目展示：中文四槽抽取中的拒判与召回权衡

面向后训练算法实习／校招。这里展示一个可以复核的实验闭环：许可与去重、CPU 上的 QLoRA SFT／DPO、固定留出与配对评测、强基线、失败归因和停止上线的决定。这里的百分比是各自注明的数据集上的结果，不代表真实用户流量。

## 30 秒讲清楚

我用 Qwen2.5-0.5B-Instruct 在 16 GB、无独立 GPU 的电脑上做中文语音请求的 `date`、`time`、`place_name`、`person` 四槽抽取。v1 SFT 在 200 条实体富集确认集上把严格实体 F1 从基座 **2.55%** 提升到 **60.43%**，但 40 条无目标句全部失败。针对拒判与召回的权衡，我完成自然频率切分、v2 SFT、v3/v4 DPO、同样本 SFT 对照与 v5／v6 两轮独立测试。**v6 在 600 条全新独立测试上把混合系统做到 62.59%，是本项目四槽抽取第一次跨过预设的绝对 F1 ≥60% 门槛**（旧 BIO 55.70%、新 BIO 61.39%、v4 DPO 单模型 42.62%、上一版 v5 混合 60.77%）。但 v6 预设的独立成功条件还要求新混合比旧 v5 混合的配对 95% 区间下界大于 0，实测 **+1.82 个百分点、区间 −3.58 至 +7.19 跨 0**，因此**总体门槛仍判为失败**，我如实报告“绝对水平达标、相对增量未被确认”，不把槽位系统称为可自动上线。

## 我做了什么

| 环节 | 可核对的实现与结果 | 算法判断 |
| --- | --- | --- |
| 数据 | [许可清单](DATA_AND_MODEL_LICENSES.md)、[v1 方案](SLOT_RESEARCH.md)、[v2 方案](reports/massive-slots-v2-design.md)记录 MASSIVE `zh-CN` 来源、原句分组、去重和数据哈希 | 先固定切分和评测规则，再看模型结果；避免近重复文本跨训练和留出 |
| SFT | [训练代码](scripts/train_sft.py)、[运行清单](reports/massive-slots-training-provenance.json)记录 4-bit QLoRA、模板、步骤和权重来源 | 训练 loss 与 JSON 合法率不能替代实体正确率及拒判能力 |
| 评测 | [v1 选择锁](reports/massive-slots-selection-lock.json)、[确认集比较](reports/massive-slots-confirmation-comparison.json)记录冻结模型、严格实体匹配、逐句全对和按原句组配对的区间 | v1 SFT 相对基座 F1 增量 **+57.88 个百分点**，95% 区间 **+50.39 至 +64.44** |
| 强基线 | [v1 报告](reports/massive-slots-study-summary.md)、[v2 报告](reports/massive-slots-v2-study-summary.md)给出同量训练数据的字符 BIO | v1 BIO F1 **68.05%**；其相对 SFT 的配对差值区间跨 0，不能声称已证实优于 SFT |
| DPO 与门禁 | [v3 方案](reports/massive-slots-v3-design.md)、[偏好审计](reports/massive-slots-v3-preference-audit.json)、[结果](reports/massive-slots-v3-study-summary.md)保留 256 个新训练组、冻结参考、64 步训练与失败门槛 | 拒判明显改善但阳性召回受损；总体 F1、阳性 F1 与可靠总体增益未过门槛，停止确认评测 |
| 独立混合验证 | [v5 选择锁](reports/massive-slots-v5-selection-lock.json)、[评测代码](scripts/evaluate_massive_slots_v5_test.py)、[600 条测试结果](reports/massive-slots-v5-study-summary.md)固定 BIO 默认、DPO 概率替换和一次性测评 | 相对 BIO 的 F1 增量经配对区间支持；绝对 60% 门槛未达，且收益属于混合系统而非 DPO 单模型 |
| 扩大监督数据 | [v6 方案](reports/massive-slots-v6-design.md)、[选择锁](reports/massive-slots-v6-selection-lock.json)、[600 条测试结果](reports/massive-slots-v6-study-summary.md)把 BIO 训练组扩到 1,600 条，冻结 DPO 与阈值 `−0.08` 重新做单次独立测试 | 新混合 F1 **62.59%**，首次通过全部绝对门槛；但预设的主要相对门槛（新混合减旧 v5 混合区间下界 >0）实测 **−3.58**，七项检查六项通过、总体失败 |

## 最有信息量的失败

v1 SFT 的 JSON 合法率为 100%，但 40 条无目标句中，34 条给出有效误报，6 条输出无效。由此可见“能输出格式”与“知道何时不抽取”是两项不同能力。v2 将开发集改为自然频率，并增加目标阳性、困难负例和纯负例，但第 160 步仍有 **126/167** 条无槽失败；同量 BIO 只有 **9/167**。v3 DPO 将无槽失败降至 **28/167**，却牺牲了阳性句 F1。这三个阶段都保留了失败结果，形成了可追踪的误差分析和下一轮实验问题。

v3 相比 v2 加入了 256 个新训练组，因此两者的差异不能只归因于 DPO 目标。我又用完全相同的 256 个正确答案，从同一个 v2 adapter 出发做了[继续 SFT 对照](reports/massive-slots-v3-matched-control-summary.md)：其开发集总体 F1 为 **49.24%**，阳性句 F1 **67.36%**，无槽失败 **78/167**。DPO 相比这个对照进一步把无槽失败降至 **28/167**，但阳性句 F1 降至 **54.75%**；总体 F1 差值 **−2.13 个百分点**，配对区间 **−8.05 至 +3.46**。这说明新增正确答案本身带来部分改善，DPO 呈现更强的拒判倾向，同时伴随召回损失。对照是在看过原开发集后设计的，属于探索性分析，不能解锁 400 条确认集，也不能单独证明损失函数的因果作用。

v4 将阳性偏好占比从 25% 提至 50%，在新开发集上阳性 F1 升到 **66.29%**，无槽失败却升至 **98/171**，总体 F1 降至 **44.79%**，按门槛停止。随后只在已看过的开发集上选择一次 BIO 加 DPO 概率门控规则，并在新抽且与全部旧组不重叠的 600 条测试上[独立验证](reports/massive-slots-v5-study-summary.md)：混合 F1 **57.21%**，相对 BIO **+7.46 个百分点**；阳性 F1 **65.78%**、无槽失败 **50/429**，所有输出均合法。**独立相对收益成立，预设总体成功门槛仍失败**，因为严格实体 F1 未到 60%。

v6 回答的是“把 BIO 监督数据扩大到 1,600 条能否抬高整个混合系统”。方案与切分在 BIO 开发集评测前固定，开发集通过后才公开并生成选择锁，再用一批与 v1–v5 全部使用或封存组不重叠的 600 条新句子做单次测试。结果：旧 BIO **55.70%**、新 BIO **61.39%**、v4 DPO **42.62%**、旧 v5 混合 **60.77%**、新混合 **62.59%**；新混合阳性 F1 **71.12%**、无槽失败 **45/433**、三项合法率各 **100%**。**七项检查六项通过，唯一失败的是预设的主要相对门槛**：新混合减旧 v5 混合的配对 95% 区间下界为 **−3.58 个百分点**。这轮的价值恰在于区分了两件事——绝对水平被推过 60%，但“新数据让系统可靠超过上一版”这个增量假设没有被确认；同时新旧混合的 BIO 模型与训练组同时变化，即使通过也只能归因于整体方案，不能说是 DPO 本身的因果提升。设计中的同 1,600 组 Qwen SFT 次级实验已启动，按[独立协议](reports/massive-slots-v6-sft-protocol.md)只评开发集；主测试报告不含任何 SFT 数字。

## 面试时可展示的三处材料

1. **切分与门禁**：[v2 冻结方案](reports/massive-slots-v2-design.md)、[选择器](scripts/select_massive_slots_v2.py)和[停止报告](reports/massive-slots-v2-study-summary.md)。说明如何防止看过确认集再调模型。
2. **训练与复现**：[SFT 代码](scripts/train_sft.py)、[DPO 代码](scripts/train_dpo.py)、[v3 复现步骤](SLOT_V3_RUNBOOK.md)和模型／数据 SHA-256。说明在 CPU 上如何做参数高效训练、冻结 DPO reference 并核查模板。
3. **指标与坏例**：[严格解析与评分](scripts/evaluate_massive_slots.py)、[配对比较](scripts/compare_massive_slots_v2.py)、[v5 独立测试](reports/massive-slots-v5-study-summary.md)及[v6 独立测试](reports/massive-slots-v6-study-summary.md)。说明为什么同时看总体 F1、阳性 F1、无槽失败和合法率，以及为什么“绝对门槛达标”不等于“增量假设成立”。

## 简历表述边界

可以写“完成 0.5B 模型的 CPU 4-bit QLoRA SFT／DPO、数据去重、配对评测与失败门禁”，并分别标注 v1 **确认集**、v2–v4 **开发集**及 v5／v6 **独立测试**数字。v5 可写“固定混合系统相对 BIO 提高 7.46 个百分点，配对区间 +2.10 至 +13.23”，同时注明绝对 60% 门槛失败；v6 可写“扩大 BIO 训练组后混合系统 F1 达 62.59%，首次跨过预设绝对 60% 门槛，但相对上一版混合的配对区间 −3.58 至 +7.19 跨 0，总体独立成功条件未通过”。不能写成 DPO 单模型提升、v6 已确认优于 v5、7B／多卡经验、线上 A/B 收益或可自动执行的槽位服务。HTTP 服务包含线性意图端点和新增的四槽人工复核端点；[服务证据](SERVING_EVIDENCE.md)说明四槽延迟尚未在空闲机器上验收。
