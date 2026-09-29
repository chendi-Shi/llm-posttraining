# 后训练算法岗位：项目证据与投递说明

更新日期：2026-09-28。本页帮助招聘方快速核对公开仓库中的证据。所有成绩均区分开发集与冻结确认集；槽位 SFT 是研究成果，未用于自动抽取服务。

## 岗位要求与项目证据

| 官方岗位要求 | 本项目可核对的证据 | 当前边界 |
| --- | --- | --- |
| 数据构造、SFT、模型版本比较与独立评测：[上海人工智能实验室大模型训练算法工程师，2026-08-13](https://www.shlab.org.cn/joinus/detail/7615234376275773734?mode=social) | [四槽预注册方案](SLOT_RESEARCH.md)、[模型选择冻结记录](reports/massive-slots-selection-lock.json)、[研究报告](reports/massive-slots-study-summary.md)记录 4-bit QLoRA、第 48/96 步选择、严格复制评测、同量 BIO 基线与负结果 | 只训练 0.5B 模型；确认集上 SFT 无槽误报高，不能宣称生产级抽取能力 |
| 轻量模型 SFT、效果与效率评测：[OPPO 大模型算法工程师，2026-07-15](https://careers.oppo.com/university/oppo/campus/post/1820?recruitType=Intern) | 在本机 CPU 上对 Qwen2.5-0.5B-Instruct 做 4-bit QLoRA SFT；[实验方案](EXPERIMENT_DESIGN.md)、[训练脚本](scripts/train_sft.py)、[测试对照](reports/massive-test-comparison.json)记录数据、设置与结果 | 只训练 0.5B 模型；不等同于 7B 或多卡训练经验 |
| SFT/DPO、训练数据与独立实验分析：[上海人工智能实验室大模型训练算法工程师，2026-08-13](https://www.shlab.org.cn/joinus/detail/7615234376275773734?mode=social) | [数据许可说明](DATA_AND_MODEL_LICENSES.md)、[SFT/DPO 脚本](scripts/train_dpo.py)、[开发集对照](reports/massive-dpo-dev-comparison.json)与 [OASST2 偏好评测](reports/oasst2_preference_eval_1epoch.json)可复核流程和负结果 | DPO 未显示可靠增益；没有奖励模型、PPO/GRPO 或分布式训练成果 |
| Bad Case 归因、版本比较、从实验到上线：[博世大模型算法工程师，校招岗位](https://jobs.smartrecruiters.com/BoschGroup/744000142551949--27-bcsc) | [分类基线报告](reports/massive-linear-test-summary.md)、[发布验收](RELEASE_READINESS.md)和 [容器部署说明](CLOUD_DEPLOY.md)区分研究模型与人工复核服务 | 尚无云服务器、真实流量或线上反馈闭环；不应称为线上服务 |
| 数据治理、Agentic RL 环境与训练：[腾讯 Code/Agent 后训练算法研究员，更新于 2026-09-08](https://careers.tencent.com/zh-cn/jobdesc.html?postId=2027548243965145088)；[上海人工智能实验室智能体后训练岗位，2026-09-11](https://www.shlab.org.cn/joinus/detail/7684140915950881062?mode=social) | 当前项目有受控数据构造、模型训练和离线评测经验 | 尚无 Agent 轨迹、可验证奖励、rollout 或 Agentic RL 实验，不以这些岗位要求作为已具备能力 |

## 已验证结果与诚实解读

- MASSIVE 1.0 `zh-CN` 四槽 JSON 抽取：**384 条**训练、**100 条**开发、**200 条**冻结确认。按开发集 F1 预定规则选第 96 步 4-bit QLoRA SFT。确认集严格实体 micro-F1：基座 **2.55%**、SFT **60.43%**、同量数据 BIO **68.05%**；SFT 相对基座的配对提升为 **+57.88 个百分点**，95% 区间 **+50.39 至 +64.44**。BIO 相对 SFT 的点估计高 7.61 个百分点，但配对区间跨 0，不能宣称可靠 F1 优势。SFT 虽有 **100%** JSON 合法率，无槽句误报仍为 **34/40**，BIO 是 **7/40**；尚不具备无人复核上线条件。数据富集目标实体，不能外推为真实流量表现。[研究报告](reports/massive-slots-study-summary.md)
- MASSIVE 1.0 `zh-CN` 意图分类：594 条 SFT 训练样本，完整官方 test 为 2,974 条。Qwen 基座准确率/macro-F1 为 **26.53%/16.15%**；QLoRA SFT 为 **29.32%/21.18%**。配对 bootstrap 的 macro-F1 差值为 **+5.03 个百分点**，95% 区间 **+3.45 至 +6.70**。这证明在该任务上有小幅适配收益，不证明通用能力改善。[完整指标](reports/massive-test-comparison.json)
- 同一任务的 DPO 在开发集上相对 SFT 的 macro-F1 差值为 **+1.35 个百分点**，95% 区间 **−2.99 至 +5.61**，未达到预设门槛，因此没有把 DPO 推进到官方 test。[开发集比较](reports/massive-dpo-dev-comparison.json)
- 事后诊断中，同样使用 594 条 SFT 训练样本的字符 TF-IDF + LinearSVC，在同一 300 条开发集上达到 **64.33%** 准确率、**63.89%** macro-F1；Qwen SFT 为 **30.67%/24.95%**。这一对照提示当前意图任务和训练配置尚不足以展示模型竞争力。该线性对照是事后诊断，不作为预注册实验的确认性结论。[匹配样本报告](reports/massive-matched-594-linear-dev.json)
- 服务端选用另一个以 **11,514 条**官方 train 样本训练的约 **11 MB** 线性模型；排除与完整 train/dev 相同文本后，官方 test 的 2,727 条样本准确率为 **82.80%**、macro-F1 为 **77.99%**。服务要求人工复核，部署包已备好，尚未在云服务器对外运行。该数字不可与 594 条 SFT 训练实验当作同量数据的模型优劣比较。[线性模型摘要](reports/massive-linear-test-summary.md)

## 简历可用表述

以下是可据仓库核对的项目描述；按实际参与范围和个人经历调整。

> 在 16 GB 内存的 CPU 环境中，完成 Qwen2.5-0.5B-Instruct 的 4-bit QLoRA SFT 与严格结构化抽取评测。基于许可明确的 MASSIVE 中文数据构造 384/100/200 条训练、开发、确认划分，按开发集预定规则选择检查点；确认集实体 F1 从基座的 2.55% 提升至 60.43%（配对 95% 区间 +50.39 至 +64.44 个百分点），并公开同量数据 BIO 基线 68.05% F1 及 SFT 无槽误报 34/40 的局限。
>
> 建立包含数据去重、模型/数据哈希、逐类指标、配对 bootstrap 和错误归因的可复现实验流程；在旧意图分类任务上将官方 test macro-F1 从基座的 16.15% 提升至 21.18%，DPO 未达到预设收益门槛并保留负结果。

面试可展示 [训练代码](scripts/train_sft.py)、[槽位研究方案](SLOT_RESEARCH.md)、[模型选择冻结记录](reports/massive-slots-selection-lock.json)和 [研究报告](reports/massive-slots-study-summary.md)。如投递时公开仓库尚未同步本轮文件，应先完成发布并核对链接，再使用上述简历描述。

## 投递定位

当前证据更适合强调 **应用型后训练、SFT 实验、训练数据和模型评测** 的岗位。研究型 Agentic RL、大规模分布式训练以及要求 7B 以上微调实绩的岗位，还需要新的真实项目证据。岗位的学历、毕业年份、工作年限及城市要求须按招聘页面和本人条件逐一核对。
