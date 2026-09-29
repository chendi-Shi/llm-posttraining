# 后训练项目展示：中文四槽抽取中的拒判与召回权衡

面向后训练算法实习／校招。这里展示一个可以复核的实验闭环：许可与去重、CPU 上的 QLoRA SFT／DPO、固定留出与配对评测、强基线、失败归因和停止上线的决定。这里的百分比是各自注明的数据集上的结果，不代表真实用户流量。

## 30 秒讲清楚

我用 Qwen2.5-0.5B-Instruct 在 16 GB、无独立 GPU 的电脑上做中文语音请求的 `date`、`time`、`place_name`、`person` 四槽抽取。v1 SFT 在 200 条实体富集确认集上把严格实体 F1 从基座 **2.55%** 提升到 **60.43%**，但 40 条无目标句全部失败。这个反例促使我重新设计自然频率留出和负例；v2 SFT 仍未通过预设门槛。v3 DPO 把开发集无槽失败从 **126/167** 降至 **28/167**，同时阳性句 F1 从 **67.01%** 降至 **54.75%**，总体 F1 改进区间跨 0。我按门槛停止，封存 400 条确认集，没有把槽位模型包装成可上线服务。

## 我做了什么

| 环节 | 可核对的实现与结果 | 算法判断 |
| --- | --- | --- |
| 数据 | [许可清单](DATA_AND_MODEL_LICENSES.md)、[v1 方案](SLOT_RESEARCH.md)、[v2 方案](reports/massive-slots-v2-design.md)记录 MASSIVE `zh-CN` 来源、原句分组、去重和数据哈希 | 先固定切分和评测规则，再看模型结果；避免近重复文本跨训练和留出 |
| SFT | [训练代码](scripts/train_sft.py)、[运行清单](reports/massive-slots-training-provenance.json)记录 4-bit QLoRA、模板、步骤和权重来源 | 训练 loss 与 JSON 合法率不能替代实体正确率及拒判能力 |
| 评测 | [v1 选择锁](reports/massive-slots-selection-lock.json)、[确认集比较](reports/massive-slots-confirmation-comparison.json)记录冻结模型、严格实体匹配、逐句全对和按原句组配对的区间 | v1 SFT 相对基座 F1 增量 **+57.88 个百分点**，95% 区间 **+50.39 至 +64.44** |
| 强基线 | [v1 报告](reports/massive-slots-study-summary.md)、[v2 报告](reports/massive-slots-v2-study-summary.md)给出同量训练数据的字符 BIO | v1 BIO F1 **68.05%**；其相对 SFT 的配对差值区间跨 0，不能声称已证实优于 SFT |
| DPO 与门禁 | [v3 方案](reports/massive-slots-v3-design.md)、[偏好审计](reports/massive-slots-v3-preference-audit.json)、[结果](reports/massive-slots-v3-study-summary.md)保留 256 个新训练组、冻结参考、64 步训练与失败门槛 | 拒判明显改善但阳性召回受损；总体 F1、阳性 F1 与可靠总体增益未过门槛，停止确认评测 |

## 最有信息量的失败

v1 SFT 的 JSON 合法率为 100%，但 40 条无目标句中，34 条给出有效误报，6 条输出无效。由此可见“能输出格式”与“知道何时不抽取”是两项不同能力。v2 将开发集改为自然频率，并增加目标阳性、困难负例和纯负例，但第 160 步仍有 **126/167** 条无槽失败；同量 BIO 只有 **9/167**。v3 DPO 将无槽失败降至 **28/167**，却牺牲了阳性句 F1。这三个阶段都保留了失败结果，形成了可追踪的误差分析和下一轮实验问题。

v3 相比 v2 加入了 256 个新训练组，因此两者的差异不能只归因于 DPO 目标。我又用完全相同的 256 个正确答案，从同一个 v2 adapter 出发做了[继续 SFT 对照](reports/massive-slots-v3-matched-control-summary.md)：其开发集总体 F1 为 **49.24%**，阳性句 F1 **67.36%**，无槽失败 **78/167**。DPO 相比这个对照进一步把无槽失败降至 **28/167**，但阳性句 F1 降至 **54.75%**；总体 F1 差值 **−2.13 个百分点**，配对区间 **−8.05 至 +3.46**。这说明新增正确答案本身带来部分改善，DPO 呈现更强的拒判倾向，同时伴随召回损失。对照是在看过原开发集后设计的，属于探索性分析，不能解锁 400 条确认集，也不能单独证明损失函数的因果作用。

## 面试时可展示的三处材料

1. **切分与门禁**：[v2 冻结方案](reports/massive-slots-v2-design.md)、[选择器](scripts/select_massive_slots_v2.py)和[停止报告](reports/massive-slots-v2-study-summary.md)。说明如何防止看过确认集再调模型。
2. **训练与复现**：[SFT 代码](scripts/train_sft.py)、[DPO 代码](scripts/train_dpo.py)、[v3 复现步骤](SLOT_V3_RUNBOOK.md)和模型／数据 SHA-256。说明在 CPU 上如何做参数高效训练、冻结 DPO reference 并核查模板。
3. **指标与坏例**：[严格解析与评分](scripts/evaluate_massive_slots.py)、[配对比较](scripts/compare_massive_slots_v2.py)及[v3 结果](reports/massive-slots-v3-study-summary.md)。说明为什么同时看总体 F1、阳性 F1、无槽失败和合法率。

## 简历表述边界

可以写“完成 0.5B 模型的 CPU 4-bit QLoRA SFT／DPO、数据去重、配对评测与失败门禁”，并分别标注 v1 **确认集**和 v2/v3 **开发集**数字。不能写成 7B／多卡经验、DPO 稳定提升、线上 A/B 收益或可自动执行的槽位服务。项目里的 HTTP 人工复核服务使用的是另一项意图分类小模型；[服务说明](SERVICE_RUNBOOK.md)与槽位研究分开。
