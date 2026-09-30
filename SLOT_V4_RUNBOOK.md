# 四槽抽取 v4：平衡偏好训练复现

本轮实验方案、样本配额与通过门槛见 [v4 预先方案](reports/massive-slots-v4-design.md)。所有数据来自 MASSIVE 1.0 `zh-CN` 官方 train，遵守 CC BY 4.0；原句、偏好对、逐题预测和模型权重不进入公开仓库。先按 [v2 步骤](SLOT_V2_RUNBOOK.md)准备原始归档、v1/v2 数据和 Qwen2.5-0.5B-Instruct 基座，再按 [v3 步骤](SLOT_V3_RUNBOOK.md)准备 v3 候选池与冻结的 v3 DPO adapter。

## 固定新切分和偏好对

```powershell
.\.venv\Scripts\python.exe scripts\prepare_massive_slots_v4.py
.\.venv\Scripts\python.exe scripts\build_massive_slots_v4_preferences.py
```

第一个命令应生成 256 条训练、250 条开发、400 条封存确认样本。manifest SHA-256 必须是 `c3bc00cadb70fc6c7a7487e19a2e23f6e57726b4b97cbfaa3582c9835395c206`。第二个命令从冻结 v2 SFT 模型生成错误回答并构造 256 对偏好数据；本次文件 SHA-256 为 `43fe0918585fa01e2b84427eff37d03966ecfd481b94dc458b6f8e16d57b037a`，其中 160 对取自模型真实错误、96 对使用固定错误替代，最长完整序列 156 token。

## 只训练一个预先指定的模型

```powershell
.\.venv\Scripts\python.exe scripts\train_dpo.py --model models/Qwen2.5-0.5B-Instruct --sft-adapter outputs/massive-slots-v2-sft-lr3e4-160 --train-file data/massive-zh/slots-v4/preferences.jsonl --output-dir outputs/massive-slots-v4-balanced-dpo-64 --max-steps 64 --max-length 256 --learning-rate 1e-5 --beta 0.1 --seed 20261001 --gradient-accumulation-steps 4 --save-steps 8
```

训练器先计算冻结 reference 的对数概率，再运行 64 个 optimizer steps。第 8、16、… 步的保存点仅供中断恢复；只评最终第 64 步。完成后检查 `outputs/massive-slots-v4-balanced-dpo-64/run_manifest.json` 的训练文件、起始 adapter 哈希与 `global_step=64`，再公布最终 adapter 的 SHA-256 和保存精度。

## 新开发集单次评测

开发集评测脚本已在生成性能分数前发布。完成训练并记录 adapter 哈希后运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_massive_slots_v4_dev.py --expected-adapter-sha256 <最终 adapter_model.safetensors 的 SHA-256>
```

该脚本对冻结 v3 DPO 和 v4 DPO 在**同一个全新 250 条开发集**分别贪心推理，输出聚合指标、配对 bootstrap 区间和预定门槛判断。逐题预测只写入忽略目录 `_tmp/`。报告为 `reports/massive-slots-v4-dev.json`，评测入口拒绝覆盖已有报告或预测。若门槛未全部通过，新确认集保持封存；不得以中途检查点或额外调整重写本轮结果。若全部通过，先冻结开发集报告、模型和推理代码哈希，再使用新 400 条确认集一次性验证。原 v2 400 条确认集始终不参与本轮模型选择。
