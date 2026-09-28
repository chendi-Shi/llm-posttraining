# MASSIVE 中文意图分类：完整训练集轻量基线（开发集）

## 范围与复现

- 数据：官方 MASSIVE 1.0 `zh-CN`，仅使用 train 11,514 条和 dev 2,033 条。**未使用 test 做训练、选择或评分。**
- 模型：字符 TF-IDF + LinearSVC，CPU 训练；从两个预设候选中按文本去重后的 dev macro-F1 选择。
- 最优候选：字符 1–3 gram、`min_df=2`、`sublinear_tf=True`、`C=1`、不做类别重加权。
- 在项目根目录用项目虚拟环境复现：`.\.venv\Scripts\python.exe scripts\evaluate_massive_linear.py --report reports\massive-linear-dev-rerun.json --predictions reports\massive-linear-dev-rerun.jsonl --model-out outputs\massive-linear-baseline-rerun.joblib`。脚本拒绝覆盖已有结果。

| 候选 | dev 全量准确率 | dev 全量 macro-F1 | 文本未见 dev 准确率 | 文本未见 dev macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 字符 1–3 gram，常规权重（入选） | 82.98% | 81.51% | 82.27% | 79.23% |
| 字符 2–4 gram，类别平衡 | 79.59% | 77.82% | 78.56% | 75.73% |

## 文本重复影响

train 与 dev 的 ID 不交叉，但共享 132 种逐字相同的请求，涉及 dev 的 144 条记录。以 NFKC、大小写折叠并去除标点/空格后的文本键检查，数量仍为 132 种和 144 条。排除这些记录后，dev 保留 1,889 条且覆盖原有全部 59 个类别。入选模型 macro-F1 从 81.51% 降至 79.23%，准确率从 82.98% 降至 82.27%。因此正式验收应使用文本隔离的数据。

## 限制

文本未见 dev 的 `cooking_query` 仅 2 条、`general_greet` 仅 1 条，两类 F1 均为 0；开发集不含 `audio_volume_other`。`general` 场景准确率只有 47.8%。模型在本次 dev 上选择，以上分数是开发阶段结果，**不能视为独立最终测评或直接上线证明**。后续应冻结模型与阈值，使用独立盲测评估真实中文请求、未知意图、拒识、逐类错误和单请求延迟。

机器可读的逐类/场景指标、候选配置及重复审计见 [完整报告](massive-linear-dev.json)，逐题结果见 [预测文件](massive-linear-dev.jsonl)。保存的模型在 `outputs/massive-linear-baseline.joblib`，已验证加载后对 2,033 条 dev 的预测与报告逐项一致。
