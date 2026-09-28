# OASST2 匿名生成盲评结果

## 评分方式

- 对 33 个未参与训练的 validation 提示，比较基座、完整一轮 SFT、完整一轮 SFT+DPO 各自生成的回答，共 99 条。
- 每题的回答标签随机打乱为 A/B/C；评分时不读取揭盲映射。
- 按切题程度、遵循上下文与指令、准确性和表达清楚度，选出最好的回答；无法区分时记并列。评分完成后再按 key 文件映射回模型。
- 这是一次单一 AI 评审，不是人工或多评审盲测。每条回答限制为最多 96 个新 token，部分较长回答因此被截断。

## 结果

| 模型 | 单独胜出 | 平分后的胜出份额 |
| --- | ---: | ---: |
| 基座 | 15 题 | 47.5% |
| SFT，一轮 | 5 题 | 30.8% |
| SFT+DPO，一轮 | 2 题 | 21.7% |
| 并列 | 11 题 | — |

平分后的份额把并列题的胜出分数均分给并列答案，三种模型的份额合计为 100%。本轮 AI 盲评没有发现 SFT 或 DPO 优于基座的证据，方向上反而偏向基座。结合 33 题的小样本、任务类型有限、单一评审及回答截断，这只是初步诊断结果，不能当作正式能力基准。

## 逐题评分

逐题首选标签和理由记录在 [oasst2_blind_generation_ratings.csv](oasst2_blind_generation_ratings.csv)。模型标签对应关系在 [oasst2_blind_generation_key.json](oasst2_blind_generation_key.json)。原始匿名回答在 [oasst2_blind_generation_eval.md](oasst2_blind_generation_eval.md)。
