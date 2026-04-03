# weibo_classification 项目文档（中文）

> 用途：把微博抓取结果做成“话题簇/KMeans + 多标签推断 + 簇命名（DeepSeek）”的端到端工作链，并给出 Web 页面与 CLI 的所有入口参数/流程说明。

## 1. 项目概览（你能用它做什么）

1. **抓取微博数据**
   - 关键词多圈层抓取：先召回 `output/unified_*.jsonl`，再从候选用户中均衡抽样写入 `input/user_ids.txt`，最后抓取用户画像与时间线（也会产生新的 `output/unified_*.jsonl`）。
   - Web 抓取单个 UID：直接抓 `weibospider` 中的 `user` + `tweet_by_user_id`，最终合并为固定文件 `output/weibo_crawl_latest.json`。
2. **推文级 KMeans 聚类（主链路）**
   - 把“单条微博正文”编码成向量：`768-d -> PCA(64) -> L2 -> silhouette 扫 k -> KMeans`
   - 把每个簇中心映射到 8 个兴趣类别（或标为 Noise）。
3. **用户兴趣画像（infer）**
   - 对用户聚合推文簇占比，并训练/调用 GBDT 多标签模型输出用户兴趣分布与解释。
4. **簇深度命名（DeepSeek）**
   - 从聚类的样本文本中抽样，调用 DeepSeek 输出中文四字标签 + 人群画像/共性感知/总结。

## 2. 关键目录与文件

### 2.1 抓取/合并产物

- `output/unified_*.jsonl`
  - Scrapy 的统一行格式：一行一条记录，包含 `user_id/text/source_type/raw/...`
- `output/weibo_crawl_latest.json`
  - 抓取合并后的 bundle（结构：`users[] -> records[]`）
  - Web 的完整抓取任务会把本次新增的 `unified_*.jsonl` 合并成它（并保留 unified_*.jsonl 以便溯源）

### 2.2 KMeans 主链路产物

- `output/kmeans_tweets_input.jsonl`
  - **步骤①**的固定导出文件：每行一条“已清洗的微博正文”（train/infer 的 `--input` 默认会优先用它）
- `output/ml_artifacts/`
  - `cluster_viz_data.json`：前端散点图与样本文本
  - `kmeans_k.json`：最佳 k 与搜索结果
  - `kmeans_tweet_model.pkl`：训练得到的 KMeans 模型
  - `tweet_pca_64.pkl`：PCA(64) 模型
  - `cluster_label_map.json`：簇 -> 兴趣标签映射
  - `gbdt_multilabel.pkl`：多标签 GBDT 模型
  - `training_pipeline.json`：训练元信息（infer 用来判断/对齐）
  - `user_cluster.json`、`metrics.json`：训练辅助文件
  - `cluster_llm_topic.json`：DeepSeek 深度命名结果（簇到人群画像/感知/总结）

### 2.3 旧的用户级流水线（历史/对照）

- `output/cleaned_user_texts.jsonl`：用户长文本（旧流程用）
- `output/text2vec_merged/`、`output/user_features_text2vec.npy`、`output/kmeans_multilabel_users.jsonl`：旧流程用

> 本项目当前 **Web 主流程**更偏向推文级 KMeans + infer，而不是旧的用户级清洗页。

## 3. 依赖安装

### 3.1 requirements.txt

`requirements.txt` 已包含主要依赖：

```txt
Scrapy==2.5.1
python_dateutil
cryptography==36.0.2
pyOpenSSL==22.0.0
Twisted==22.10.0
fastapi
uvicorn
jinja2
python-multipart
scikit-learn
jieba
numpy
sentence-transformers
ijson
```

### 3.2 创建虚拟环境并安装

PowerShell（示例）：

```powershell
cd F:\Desktop\weibo_classification
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. 模型下载/缓存办法（text2vec）

本项目默认使用：
- `shibing624/text2vec-base-chinese`（SentenceTransformer）

### 4.1 在线自动下载（最简单）

当你运行需要编码文本的步骤（如 KMeans 聚类、infer）时，SentenceTransformer 会自动从 HuggingFace 下载并缓存到本机。

你可以用镜像加速（建议）：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HUGGINGFACE_HUB_BASE_URL="https://hf-mirror.com"
```

### 4.2 手动离线下载（推荐）

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub
.\.venv\Scripts\python.exe -m huggingface_hub.snapshot_download `
  --repo-id shibing624/text2vec-base-chinese `
  --local-dir models/text2vec-base-chinese
```

之后在主流程里你可以让 `train_tweet_topic.py` / 相关脚本优先加载本地 `models/text2vec-base-chinese`（脚本里有本地优先逻辑）。

## 5. Web 页面入口（路由/目的）

Web 启动：
```powershell
.\.venv\Scripts\python.exe run_web.py
```
默认：
- `http://127.0.0.1:8000/`

页面：

1. `/`：总览（显示 profiles 数、主话题统计、数据文件状态、清理面板）
2. `/crawl`：抓取单个 UID（按钮提交到 `POST /api/crawl_weibo`）
3. `/keyword-crawl`：关键词多圈层抓取说明页（建议在终端运行 `multidim_crawl.py`）
4. `/train`：**主链路**（步骤①导出 -> 步骤②聚类 -> 步骤③ infer -> 步骤④一键 -> 步骤⑤ DeepSeek 命名）
5. `/profiles`：用户画像表
6. `/users/{user_id}`：用户兴趣详情页（当前展示推断摘要等）
7. `/users/{user_id}/weibo-sources`：原始微博列表页（显示正文/链接；数据来源 unified_*.jsonl + weibo_crawl_latest.json）
8. `/personal-interest`：占位页

旧 `/clean` 已重定向到 `/train`。

## 6. Web 主链路：详细流程（步骤①～⑤）

### Step 1：导出 KMeans 输入（步骤①）

入口：
- Web `/train` 点击 **① 导出 kmeans_tweets_input.jsonl**
- 调用接口：`POST /api/kmeans_prep`

参数：
- `source_file`（可空；空则优先使用 `output/kmeans_tweets_input.jsonl`，否则 bundle/unified 兜底）
- `min_tweet_chars`（默认 5）

产物：
- `output/kmeans_tweets_input.jsonl`
  - 每行 JSON：`{"user_id","text","source_type","item_id",...}`
  - `text` 已按 KMeans 嵌入前规则清洗

### Step 2：聚类（步骤②）

- 接口：`POST /api/cluster_text2vec`
- 内部运行：`python -m app_pipeline.train_tweet_topic ... --viz-only`

产物（至少）：
- `output/ml_artifacts/cluster_viz_data.json`
- `output/ml_artifacts/kmeans_k.json`

### Step 3：生成画像（步骤③）

- 接口：`POST /api/infer_text2vec`
- 内部运行：`python -m app_pipeline.infer --input ... --model-dir ... --output ...`

产物：
- `output/user_interest_profiles.json`

### Step 4：一键聚类+画像（步骤④）

- 接口：`POST /api/cluster_and_infer_text2vec`
- 内部先跑 `train_tweet_topic`（生成 training_pipeline.json 等）再跑 infer

产物：
- `output/ml_artifacts/*`（更完整训练产物）
- `output/user_interest_profiles.json`

### Step 5：DeepSeek 簇命名（步骤⑤）

- 接口：`POST /api/tweet_cluster_llm/run`
- 产物：
  - `output/ml_artifacts/cluster_label_map.json`
  - `output/ml_artifacts/cluster_llm_topic.json`

## 7. Web API 完整列表（入参/作用）

你可以以 `app_web/app.py` 为准。

重要 API：

1. `POST /api/crawl_weibo`
   - `user_id`（必填）
   - `cookie`（必填）
   - `max_pages`（默认 1）
   - `download_delay`（默认 0.2）
   - `concurrent_requests`（默认 8）
   - `retry_times`（默认 2）
2. `POST /api/kmeans_prep`（步骤①）
   - `source_file`（可空）
   - `min_tweet_chars`（默认 5）
3. `POST /api/cluster_text2vec`（步骤②）
   - `input_file`（可空）
   - `k_min`（默认 8）
   - `k_max`（默认 15）
4. `POST /api/infer_text2vec`（步骤③）
   - `input_file`（可空）
5. `POST /api/cluster_and_infer_text2vec`（步骤④）
   - `input_file`（可空）
   - `k_min`（默认 8）
   - `k_max`（默认 15）
6. `POST /api/tweet_cluster_llm/run`（步骤⑤）
   - `force`（0/1）

状态与轮询：
- `GET /api/jobs/{job_id}`
- `GET /api/workspace`
- `GET /api/cluster_preview`

清理：
- `POST /api/cleanup_output`
  - `mode`: `selective|full`
  - `keep_ml_artifacts`: 是否保留 `output/ml_artifacts`
  - 以及多种 `del_*` 开关

## 8. 终端 CLI（可直接运行）

### 8.1 多维关键词抓取：`multidim_crawl.py`

推荐命令：

```powershell
.\.venv\Scripts\python.exe multidim_crawl.py `
  --keywords "手机评测,新能源车,政治,局势,大模型,基金,ETF,游戏,动漫,旅游,美食,数码,摄影,战争,汽车" `
  --max-pages 5 `
  --per-keyword-limit 200 `
  --total-limit 1000 `
  --append-user-ids
```

参数：
- `--keywords`（必填，逗号分隔）
- `--max-pages`（默认 5）
- `--split-by-hour`（可选：启用 `WEIBO_SPLIT_BY_HOUR=true`）
- `--start-time` / `--end-time`（可选：时间范围）
- `--per-keyword-limit`（默认 100）
- `--total-limit`（默认 1000；最终写入 `input/user_ids.txt` 的总用户数）
- `--download-delay`（默认 0.2；传给 `WEIBO_DOWNLOAD_DELAY`）
- `--concurrent-requests`（默认 8；传给并发）
- `--retry-times`（默认 2）
- `--seed`（默认 42）
- `--append-user-ids`（flag；追加 user_ids，而不是覆盖）
- `--merge-after`（可选）：抓取结束后把**本次新增的** `output/unified_*.jsonl` 合并成 `output/weibo_crawl_latest.json`
- `--merge-output`（可选）：合并输出路径（默认 `output/weibo_crawl_latest.json`）

### 8.2 其它脚本

- `python -m app_pipeline.merge_unified_to_bundle --help`
- `python -m app_pipeline.train_tweet_topic --help`
- `python -m app_pipeline.infer --help`
- `python -m app_pipeline.cluster_llm_labels --help`
- `python -m app_pipeline.kmeans_multilabel --help`

这些脚本的参数可以直接 `--help` 查看，本文档在此不重复粘贴全量参数（避免版本漂移时文档过时）。

## 9. 待实现功能（当前占位）

- `/personal-interest`：占位页
- 部分训练/推断按钮在 UI 文案上可能与 CLI 实际 `--viz-only`/输入默认存在差异（以代码逻辑为准）

## 10. 故障排查（常见问题）

1. **抓取后页面看不到微博**
   - 详情页（`/users/{uid}`）不展示完整原始列表，原始列表在 `/users/{uid}/weibo-sources`
   - 原始列表页会扫描 `output/unified_*.jsonl` 与 `output/weibo_crawl_latest.json`
2. **infer 找不到 training_pipeline.json**
   - 通常需要走“步骤④一键聚类+生成画像”或确保你运行的聚类不是 `--viz-only`

---
（文档到此结束）

