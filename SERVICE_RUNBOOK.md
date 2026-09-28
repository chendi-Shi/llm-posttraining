# 本地意图候选服务

这个服务把 MASSIVE 中文意图分类模型暴露为本机 HTTP 接口，供人工审核候选标签。默认后端是用官方完整训练集训练的字符 TF-IDF + LinearSVC；原有 Qwen SFT adapter 可用 `--backend sft` 选择。线性模型在官方完整 test 上准确率 83.76%、macro-F1 79.30%；排除与 train/dev 重复的短语后为 82.80%/77.99%。同一 test 曾在早先 SFT 研究中被查看过，因此该结果是冻结模型的确认，尚无真实场景或范围外输入验收。两个后端都没有经过置信度或拒判校准。

接口始终返回 `intent: null`、`abstained: true`、`review_required: true`、`auto_execute: false`；调用方不能用候选标签自动触发闹钟、邮件、订单等动作。

## 启动

在 `llm-posttraining` 目录，用现有的项目虚拟环境启动默认线性后端：

```powershell
.\.venv\Scripts\python.exe scripts\serve_massive.py --backend linear --port 8765
```

`--backend linear` 可省略。默认读取 `outputs/massive-linear-baseline.joblib`，并在 `joblib.load` 前核对冻结模型 SHA-256 `fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e`。加载自训模型可用 `--linear-model` 指定文件，并用 `--expected-linear-sha256` 显式提供其校验值；**先核对该模型的训练报告和指标，再改预期哈希**。只加载可信的本地 joblib 文件，因为 joblib 反序列化可执行代码。启动输出包含实际 SHA-256 和 `model_version`。

如需运行原有 SFT 后端：

```powershell
.\.venv\Scripts\python.exe scripts\serve_massive.py --backend sft --port 8765
```

SFT 默认加载 `models/Qwen2.5-0.5B-Instruct`、`outputs/massive-sft` 和 `data/massive-zh/intents.json`，路径可分别用 `--model`、`--adapter`、`--labels` 指定。服务默认监听 `127.0.0.1`。容器内可显式用 `--host 0.0.0.0` 接收容器网络请求，但宿主机端口应只映射到 `127.0.0.1`；外网访问需经受控反向代理提供身份验证和 TLS，不能直接公开此无认证接口。当前本机验证环境为 Python 3.10.21；线性后端用 joblib 1.6.0、NumPy 1.26.4、scikit-learn 1.7.2、SciPy 1.15.3，对应 `requirements-serve.txt`。SFT 另用 PyTorch 2.14.0+cpu、accelerate 1.15.0、bitsandbytes 0.50.2、PEFT 0.21.0、Transformers 5.17.0，对应 `requirements-serve-sft.txt`（PyTorch 需单独安装）。模型与数据许可需随实际发布包单独核对。

若端口 8765 已被其他程序占用，运行 `--port 0` 让系统选择空闲端口，并从启动输出的 `listening` 字段读取实际地址。本机验收时 8765 已被其他服务占用，使用此方法成功启动。

## 接口

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health/live
Invoke-RestMethod http://127.0.0.1:8765/health/ready
Invoke-RestMethod http://127.0.0.1:8765/v1/intents -Method Post -ContentType 'application/json' -Body '{"text":"明天天气如何"}'
```

`POST /v1/intents` 只接受 `{"text":"..."}`。成功响应示例：

```json
{
  "intent": null,
  "candidate_intent": "weather_query",
  "abstained": true,
  "reason": "confidence_not_calibrated",
  "review_required": true,
  "auto_execute": false,
  "model_version": "linear-fa9f1cb3c724",
  "elapsed_ms": 1234.5
}
```

服务默认限制请求体 4,096 字节、文本 500 字符；SFT 另限制完整提示 512 token；超长返回 413。推理一次只处理一条请求，繁忙时返回 429。默认响应期限为 30 秒，超时返回 504 且就绪探针临时返回 503，直到后台推理实际结束。Python 线程无法被安全地强制停止；若模型调用长期卡住，应重启服务进程。参数可用 `--max-body-bytes`、`--max-chars`、`--max-input-tokens` 和 `--timeout-seconds` 调整。服务不记录用户输入正文。

## 本地验证

不加载模型、无需下载权重的接口测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_serve_massive.py" -v
```

测试覆盖健康检查、候选标签的人工审核约束、输入校验、长度限制、繁忙返回、超时、就绪恢复、线性模型哈希以及与训练一致的文本正规化。要核验真实权重，启动服务后调用上述三个接口，并核对 `model_version` 与启动日志一致。上线前仍需真实场景质量评测、置信度/拒判校准、单请求延迟及并发压测，并为发布环境固定完整依赖与模型许可材料。

2026-09-28 已在本机加载真实基座、SFT adapter 和标签文件完成一次请求：`/health/ready` 返回 200；“明天天气如何”返回候选 `weather_query`，`intent=null`、`review_required=true`、`auto_execute=false`，服务报告本次推理耗时 1,265 毫秒。组合版本为 `fdf756fa7fcb-296df00ceadc-69f796cfd277`。这是一条烟雾检查，不代表延迟分布或准确率验收。

2026-09-28 已在本机加载真实线性模型并通过 HTTP 请求：`/health/ready` 返回 200；“明天天气如何”返回候选 `weather_query`，`intent=null`、`review_required=true`、`auto_execute=false`，模型版本为 `linear-fa9f1cb3c724`。这也是单条烟雾检查，不代表真实场景验收。
