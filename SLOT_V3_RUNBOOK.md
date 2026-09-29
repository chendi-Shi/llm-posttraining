# 四槽抽取 v3：训练样本偏好纠错

本轮使用 [预先冻结的方案](reports/massive-slots-v3-design.md)。候选样本只来自 MASSIVE 1.0 `zh-CN` 官方 train，排除 v2 三切分及其他已用文本组。v2 的 400 条确认集在开发集门槛全部达成之前不能读取或评分。原句、偏好对、逐题预测和权重均位于 Git 忽略目录，公开仓库仅保留代码、哈希与聚合报告。

## 1. 构造全新训练组和偏好对

先按 [v2 复现步骤](SLOT_V2_RUNBOOK.md)准备 MASSIVE 归档、v1/v2 数据和 v2 第 160 步 SFT adapter。执行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v3_preferences.py
.\.venv\Scripts\python.exe scripts\build_massive_slots_v3_preferences.py
.\.venv\Scripts\python.exe scripts\audit_massive_slots_v3_preferences.py
```

候选池应为 576 个与 v2 不重叠的组，SHA-256 `2fc7e03e67417b4817bfcc3a557d0f1e950015e141e8cfdf26664214bd2e0c0f`。固定选择四类目标槽位各 16 个阳性、96 个困难负例和 96 个纯负例，合计 256 对。冻结的 SFT 模型先对这些全新训练句生成原始回复；错误回复优先作为 `rejected`，模型答对或过长才按固定规则构造错误回复。偏好数据 SHA-256 应为 `310c26b37fd7eee7cbfb4876def2d398fcd5f0d188a24da20b24aa474471c450`。

[偏好对聚合审计](reports/massive-slots-v3-preference-audit.json)确认 187 对使用模型错误、69 对使用固定替代错误；230 个 `rejected` 可解析，26 个不可解析。所有 `chosen` 都是可复制的严格四槽 JSON，所有 `rejected` 都与正确答案不同；最长完整序列 145 token，低于固定上限 256。审计再次确认与 v2 三切分的组交集为零。

## 2. 从冻结 SFT 适配器继续 DPO

先用少量训练样本做一步程序检查，不使用开发集。正式训练命令固定为：

```powershell
.\.venv\Scripts\python.exe scripts\train_dpo.py --model models/Qwen2.5-0.5B-Instruct --sft-adapter outputs/massive-slots-v2-sft-lr3e4-160 --train-file data/massive-zh/slots-v3/preferences.jsonl --output-dir outputs/massive-slots-v3-dpo-64 --max-steps 64 --max-length 256 --learning-rate 1e-5 --beta 0.1 --seed 20260930 --gradient-accumulation-steps 4 --save-steps 32
```

TRL 会把起始 SFT adapter 复制为冻结 `ref`，再优化默认 adapter。训练输出记录数据、基座、tokenizer、起始与参考 adapter 的哈希和参数。第 32 步检查点只用于故障诊断；模型选择只依据完成的第 64 步。CPU 训练可能需要较长时间。

本次保存的最终默认 adapter 为 `bfloat16`，起始 SFT 与冻结 `ref` 均为 `float32`。评测前已在 [v3 方案的执行记录](reports/massive-slots-v3-design.md)公开此精度差异；运行清单中的 `training_seconds` 不含参考分数预计算。

## 3. 固定开发集门槛

完整训练结束后执行一次：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v3_dev.py
```

该入口只接受第 64 步训练清单与冻结资产，读取 v2 的 250 条开发集及其已保存的 v2 SFT 预测，输出聚合报告和本地逐题预测。全部门槛见 [v3 设计](reports/massive-slots-v3-design.md)：JSON／结构／复制合法率各不低于 95%、无槽失败率不高于 30%、总体严格实体 F1 不低于 55%、阳性子集 F1 不低于 60%，且相对 v2 SFT 的配对 F1 增量和无槽失败下降的 95% 区间下界均大于 0。

**任何一项未达标即停止。**仅当全部达标，先冻结开发集报告、逐题预测、数据、模型和推理代码的 SHA-256，再另行执行一次确认集推理。确认集不得用于改偏好样本、步数或门槛。v3 增加了新训练句，不能将 v3 与 v2 差异单独归因为 DPO 目标的效果。

## 4. 事后同样本继续 SFT 对照

v3 DPO 未通过门槛后，才设计[同样本对照](reports/massive-slots-v3-matched-control-design.md)，因此它是**探索性开发集分析**，不能解锁确认集。先从已冻结偏好对中逐项提取 `prompt` 和正确 `chosen`，核对全部 256 条的模板和 completion 掩码：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v3_matched_sft.py
.\.venv\Scripts\python.exe scripts\check_sft_template_mask.py --train-file data/massive-zh/slots-v3/matched-sft.jsonl --all --max-length 256
```

派生训练文件 SHA-256 应为 `8de93fdabe1b580def8bb10bef6e6cdb05942ff621bc29c0e7a34db1f0eed25c`。首次训练在第 31 步后意外退出且没有恢复点；下面是按公开偏离记录从头完成的重跑命令，唯一训练计划变更是每 8 步保存完整检查点：

```powershell
.\.venv\Scripts\python.exe scripts\train_sft.py --model models/Qwen2.5-0.5B-Instruct --init-adapter outputs/massive-slots-v2-sft-lr3e4-160 --train-file data/massive-zh/slots-v3/matched-sft.jsonl --output-dir outputs/massive-slots-v3-matched-sft-64-restart1 --max-steps 64 --max-length 256 --learning-rate 1e-5 --seed 20260930 --gradient-accumulation-steps 4 --save-steps 8
```

最终 adapter SHA-256 为 `97aa3eaf5ea19442f0c50bdd9048807b9de9a931fe86f17a11eb578e1e3343fc`，保存精度为 `torch.bfloat16`。这些指纹已在开发集推理前公开。固定对照评测命令如下；入口会校验训练清单、训练文件、基座和 tokenizer、adapter 及冻结参考预测的哈希，并拒绝重复写入已有报告：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v3_matched_control.py --expected-adapter-sha256 97aa3eaf5ea19442f0c50bdd9048807b9de9a931fe86f17a11eb578e1e3343fc --expected-adapter-dtype torch.bfloat16
```

已完成的[聚合结果](reports/massive-slots-v3-matched-control-summary.md)与[机器可读报告](reports/massive-slots-v3-matched-sft-dev.json)公开；逐题预测、原始数据和权重仍只保留本地。重新运行前需在独立目录重建数据与模型，不应覆盖本轮报告或拿 400 条确认集调整结论。
