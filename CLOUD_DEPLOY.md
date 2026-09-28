# 云服务器部署：人工复核候选服务

本部署包在 Linux 云服务器上提供 MASSIVE 中文意图候选。接口始终返回 `intent: null`、`review_required: true`、`auto_execute: false`；它只供授权人员复核，不执行用户请求对应的动作。模型尚未针对真实业务输入、未知意图和拒判阈值完成验收。

## 包内内容与来源

Docker 构建从 [Amazon Science 官方 MASSIVE 1.0 归档](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz)下载数据，校验 SHA-256 `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8`，再运行 `scripts/evaluate_massive_linear.py`。构建会确认选中的候选是 `char_1_3`、训练报告没有使用 test，并核对模型文件哈希。训练阶段可供审计；最终运行镜像只复制约 11 MB 的模型、线性服务代码及官方数据许可和署名文件，不含原始数据、测试文本、开发集报告或 Qwen 权重。

MASSIVE 数据采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；构建从官方归档中提取 `LICENSE` 和 `NOTICE.md`，放在镜像 `/app/attribution/`，另说明本项目的模型训练用途。详见[项目数据与模型许可说明](DATA_AND_MODEL_LICENSES.md)。

## 从干净检出审计构建

需要 Linux 主机、Docker Engine、Docker Compose v2，以及构建时可访问官方数据下载地址和 Python 包源。本项目应是独立检出目录，且本地 Docker 构建上下文已由 `.dockerignore` 限定为四个脚本、Dockerfile 和线性后端固定依赖。

先构建**仅用于审计、不可作为服务镜像发布**的 `trained` 阶段：

```sh
docker build --target trained -t massive-intent-review:trained .
docker run --rm --entrypoint python massive-intent-review:trained -c 'import json; r=json.load(open("/app/reports/massive-linear-dev.json")); print("selected:",r["selected_candidate"]); print("source:",r["source_sha256"]); print("test_used:",r["test_used"]); print([(x["name"],x["phrase_disjoint_dev"]["macro_f1"],x["phrase_disjoint_dev"]["accuracy"]) for x in r["candidates"]])'
docker run --rm --entrypoint sha256sum massive-intent-review:trained /app/outputs/massive-linear-baseline.joblib
```

已核对的本机训练报告选中 `char_1_3`，其去重开发集 macro-F1 为约 `0.7923`、准确率约 `0.8227`。构建脚本锁定候选名称，但不同操作系统或底层库可能产生不同的 joblib 字节。服务原有本机模型哈希为 `fa9f1cb3c72496060492703087eb1a5fb558e35d276e4049cd81fefbe07fe13e`。**如云服务器新哈希不同，先核对上面的训练报告与实际哈希，再把经过复核的 64 位小写哈希显式传给正式构建；不能只为让构建通过而复制输出。**

```sh
EXPECTED_MODEL_SHA256=<经复核的模型SHA-256> docker compose build
docker compose up -d --no-build
docker compose ps
```

`compose.yaml` 只将服务发布到宿主机 `127.0.0.1:8765`。服务在容器内使用 `--host 0.0.0.0` 让 Docker bridge 连接它；容器以非 root 用户、只读文件系统运行，且不挂载数据或模型目录。正式构建会在镜像内写入经过核对的模型哈希，启动时服务仍会在 `joblib.load` 前再次核验该哈希。

在服务器本机检查：

```sh
curl --fail --silent --show-error http://127.0.0.1:8765/health/live
curl --fail --silent --show-error http://127.0.0.1:8765/health/ready
curl --fail --silent --show-error -H 'Content-Type: application/json' -d '{"text":"明天天气如何"}' http://127.0.0.1:8765/v1/intents
docker compose logs --tail=100 intent-review
```

第三个响应应包含 `candidate_intent`，并保持 `intent: null`、`review_required: true`、`auto_execute: false`。健康检查通过只说明服务可用，不代表真实场景分类质量已验收。

## 接入真实访问

当前容器端口不对公网开放。要让审核人员通过互联网访问，仍需云服务器信息、域名与 DNS、HTTPS 证书、明确的访问控制方式和凭据。反向代理应部署在同一主机，只连接 `127.0.0.1:8765`，对包括健康接口在内的公开入口执行 TLS 和身份验证；防火墙不要开放 8765。凭据保存在云端密钥服务或主机受保护文件中，不提交 Git。

在拿到这些环境信息并完成带认证的端到端测试前，本部署包只提供服务器本机可访问的人工复核接口。自动执行指令需要另外完成真实请求、未知意图、拒判、延迟和误判成本验收。
