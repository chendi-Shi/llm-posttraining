# MASSIVE 中文意图分类：锁定线性模型测试确认

模型先在官方 train（11,514 条）训练，并在 dev 上从两个预设候选中锁定 `char_1_3`。本报告随后只加载保存的模型，对官方完整 test（2,974 条）运行了一次推理。**同一 test 在此前的 SFT 研究中已被查看过，因此本次属于新基线确认，不是完全未见过的盲测。** 本次没有训练、修改候选或调整阈值。

| 范围 | 样本 | 准确率 | Macro-F1 | 测试类别 |
| --- | ---: | ---: | ---: | ---: |
| 官方完整 test | 2,974 | 83.76% | 79.30% | 59 |
| 排除与完整 train/dev 重复文本 | 2,727 | 82.80% | 77.99% | 59 |

test 中有 223 条与完整 train 的请求逐字相同，55 条与 dev 逐字相同，去重并集为 247 条。以 NFKC、大小写折叠及去标点/空格后的文本键核对，数量相同。排除这些记录使准确率降低 0.96 个百分点，macro-F1 降低 1.31 个百分点。

去重后最弱的类别包括 `general_greet`（仅 1 条，F1=0）、`music_settings`（6 条，F1=0.20）和 `iot_hue_lighton`（3 条，F1=0.25）。`general` 场景准确率为 57.1%，`music` 为 65.8%。这些低支持数类别需要更多独立样本；本次闭集测评也没有评估未知意图拒识。

锁定模型 SHA-256：`fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e`；文件大小 11,180,878 字节。本机运行中模型载入 0.1295 秒，2,974 条批量向量化及预测 0.1099 秒，包含官方归档读取和评分的脚本内部总耗时 3.4111 秒。这些是批处理耗时，不代表在线单条请求的延迟。

逐类和场景指标、重复审计、来源校验及运行耗时见 [机器可读报告](massive-linear-test.json)；逐题预测见 [预测文件](massive-linear-test.jsonl)。只读复现命令：`.\.venv\Scripts\python.exe scripts\evaluate_massive_linear_test.py --report reports\massive-linear-test-rerun.json --predictions reports\massive-linear-test-rerun.jsonl`。脚本验证模型哈希并拒绝覆盖现有结果。
