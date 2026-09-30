# 四槽抽取 v5：固定混合系统的独立测试复现

实验规则、探索性阈值来源和预设成功门槛见 [v5 设计](reports/massive-slots-v5-design.md)。在已有 MASSIVE 1.0 `zh-CN` 归档、v1–v4 数据清单、冻结 v2 字符 BIO 模型与 v4 DPO adapter 的环境中，先生成只保留本地的 600 条测试集：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v5_test.py
```

清单 SHA-256 应为 `e4531ab27746a0cfe21dc0219331f5cc8237e614bb4c7090b355000fdce1d453`，测试 JSONL SHA-256 应为 `afd5655d9d7e7092ae3c7798179578fc93c3e5d4a88939e9fbfb1d5978459b3c`。先核对[公开选择锁](reports/massive-slots-v5-selection-lock.json)的 SHA-256 `d49516f8b9fe5a3b7b784a092fe7e975d8f4125249539ddad8d699c31296fd23`、BIO 和 DPO 模型哈希及代码指纹，再只运行一次：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v5_test.py --unlock-test --expected-selection-lock-sha256 d49516f8b9fe5a3b7b784a092fe7e975d8f4125249539ddad8d699c31296fd23
```

入口会先拒绝错误锁或已有结果，再读取测试原句。BIO 默认输出；有效、原文可复制的非空 v4 DPO 回复，只有完整 assistant turn 的平均 token 对数概率达到固定阈值 `−0.08` 才替换 BIO。测试中不重训、不搜索阈值、不修复非法 JSON。报告 `reports/massive-slots-v5-test.json` 和[结论](reports/massive-slots-v5-study-summary.md)为公开聚合结果；逐题预测与概率保留在 `_tmp/`，原始数据和权重保留在 Git 忽略目录。评测入口拒绝覆盖已有结果，复跑需在隔离的新检出目录重建同一文件与权重。
