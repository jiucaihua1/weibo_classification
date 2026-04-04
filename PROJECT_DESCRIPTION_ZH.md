# 微博兴趣画像项目说明（详细版）

你的项目是一个“微博兴趣画像工作链 + Web 看板”的端到端系统：从抓取数据开始，经过清洗与建模（KMeans 聚类 + 多标签推断），再把聚类与分类结果解释成“用户兴趣画像”，最后在网页上浏览与溯源。你强调的两点——**轻量级 BERT（text2vec）做语义向量**、**GBDT 做多标签分类**——是这条链路的核心组成部分。

## 1) 系统做什么（输入到输出）

输入主要来自微博抓取：
- `output/unified_*.jsonl`：Scrapy 的“统一格式”数据（每行一条记录），包含 `user_id / text / source_type / raw/...`
- `input/user_ids.txt`：从关键词结果抽样得到的目标 UID 池
- `output/weibo_crawl_latest.json`：把抓取结果合并成 bundle 结构（`users[] -> records[]`），作为训练/推断可复用的数据汇总

输出是模型与解释结果：
- `output/kmeans_tweets_input.jsonl`：训练主链路的固定导出输入（步骤①）
- `output/ml_artifacts/*`：KMeans / PCA / GBDT / 簇标签映射等模型与辅助文件
- `output/user_interest_profiles.json`：用户画像表（前端的核心展示数据）
- `output/ml_artifacts/cluster_llm_topic.json`、`cluster_label_map.json`：DeepSeek 对簇的中文四字命名与解释（步骤⑤）

## 2) 主要模块分布（项目分层结构）

### 2.1 抓取层：`weibospider/`
- `weibospider/run_spider.py`：按 mode 调度不同 spider
- `weibospider/spiders/*`：具体抓取逻辑（`user`、`tweet_by_user_id`、`tweet_by_keyword` 等）
- 产出：`output/unified_*.jsonl` 以及用户/微博抓取相关中间产物

#### `multidim_crawl.py` 与 `run_spider.py` 的分工（终端关键词整条链）

- **`multidim_crawl.py` 是编排脚本**：按固定顺序**多次**调用 `weibospider/run_spider.py`，串起整条链路：**关键词召回 → 从 unified 抽 UID → 写 `input/user_ids.txt` → 用户资料 → 用户时间线**。中间「解析 unified、均衡抽样、维护 UID 池」由 `multidim_crawl.py` 自己在磁盘上完成，不经过 `run_spider.py`。
- **`run_spider.py` 是单次爬取启动器**：每次进程只跑**一种** spider（由命令行 `mode` 决定），`process.start()` 阻塞到本次爬取结束即退出；数据落盘依赖 `settings.py` 中的 `pipelines.JsonWriterPipeline`，写入 `output/unified_*.jsonl`（及配套的 `user_aggregate_*.json` 等）。

二者关系：**调用方 / 被调用方**——`multidim_crawl.py` 用子进程在 `weibospider` 工作目录下执行 `python run_spider.py <mode>`；Web 抓取（`app_web/app.py`）同样通过子进程调用 `run_spider.py`，但通常不跑 `multidim_crawl.py` 里的「先关键词再写 user_ids」那一段编排。

### 2.2 算法与流水线：`app_pipeline/`
- `kmeans_prep.py`：步骤① 导出 `kmeans_tweets_input.jsonl`
- `train_tweet_topic.py`：步骤② 聚类 + 训练（包含 GBDT）
- `infer.py`：步骤③/④ 生成用户画像（调用 GBDT 做多标签预测）
- `cluster_llm_labels.py`：步骤⑤ 簇深度命名（DeepSeek）
- `merge_unified_to_bundle.py`：把多个 unified 合并为 `weibo_crawl_latest.json` 风格 bundle

#### `tweet_text_normalize.py` 与 `feature_tweet_embeddings.py`（分工与依赖）

- **`tweet_text_normalize`** 定「字面上算什么、怎么洗」：例如 `text_is_retweet`（转发占位是否过滤）、`clean_tweet_for_encode`（链接/@/话题/emoji 等与白名单清洗）。不读磁盘、不加载向量模型；`kmeans_prep` 等也会复用同一套规则，避免与嵌入链路「各洗各的」。
- **`feature_tweet_embeddings`** 定「从哪读数据、怎么过滤、怎么变成向量」：经 `data_io` 读 bundle 或 `unified_*.jsonl`，做记录级过滤（如 `source_type`、大 V、过短正文等），再用 text2vec 对正文编码得到句向量。

**依赖关系：**`feature_tweet_embeddings` **依赖** `tweet_text_normalize`。在 `feature_tweet_embeddings._append_tweet_from_record` 中顺序为：先 `text_is_retweet`，再 `clean_tweet_for_encode`；**只有通过这两步的文本才会进入编码**。

### 2.3 Web 服务与任务编排：`app_web/`
- `run_web.py`：启动 FastAPI（`app_web/app.py:app`）
- `app_web/app.py`：路由页面 + API + 后台任务线程 + 轮询 job 日志
- `app_web/templates/*.html`：`/train`、`/profiles`、`/users/{id}` 等展示页面

### 2.4 终端总控脚本：根目录
- `multidim_crawl.py`：把「关键词多圈层抓取 → 抽样 UID → 抓用户资料 → 抓用户时间线」串联为一体（实现细节与和 `run_spider.py` 的关系见 **§2.1** 小节末段）

## 3) 主链路怎么跑（Web「训练 / 画像」步骤①～⑤）

### 步骤①：导出 KMeans 输入（`kmeans_prep`）
- 入口：Web `/train` 调接口 `POST /api/kmeans_prep`
- 执行：`app_pipeline/kmeans_prep.py`
- 做的事：从 bundle/unified 抽取并清洗“单条微博正文”，写成每行一条的 `output/kmeans_tweets_input.jsonl`
- 产物：`output/kmeans_tweets_input.jsonl`

补充说明：
- 抓取原始数据（bundle 或 unified）整理成可直接给聚类/推断使用的标准微博输入文件。
- 读取输入源（支持两类），逐条过滤并清洗微博。
- 统一输出格式：每条写成一行 JSON，核心字段有：
  - `user_id`
  - `text`（清洗后的正文）
  - `source_type`
  - `item_id`
  - 以及部分保留字段（如 `created_at`、`crawl_time`、`spider`）。

### 步骤②：推文级聚类（`train_tweet_topic`）
- 入口：Web 调接口 `POST /api/cluster_text2vec`
- 执行：`app_pipeline/train_tweet_topic.py`
- 做的事：
  - 用轻量级 text2vec 语义向量化推文（BERT 表征）
  - `768-d -> PCA(64) -> L2 -> silhouette 扫 k -> KMeans`
  - 把簇中心映射到 8 个兴趣类（或标为 Noise）
  - 基于“用户的推文簇占比”构造多标签训练目标
  - 训练 **GBDT 多标签分类器**（GBDT 是分类器核心）
- 产物（关键）：
  - `output/ml_artifacts/cluster_viz_data.json`
  - `output/ml_artifacts/kmeans_k.json`
  - `output/ml_artifacts/kmeans_tweet_model.pkl`
  - `output/ml_artifacts/tweet_pca_64.pkl`
  - `output/ml_artifacts/cluster_label_map.json`
  - `output/ml_artifacts/gbdt_multilabel.pkl`
  - `output/ml_artifacts/training_pipeline.json`

### 步骤③：生成用户画像（`infer`）
- 入口：Web 调接口 `POST /api/infer_text2vec`
- 执行：`app_pipeline/infer.py`
- 做的事：
  - 读取步骤②产物
  - 对每个用户聚合推文向量并计算“话题占比/解释证据”
  - 调用 **GBDT 多标签模型**输出每个兴趣标签的概率/正负预测
  - 形成每个用户最终画像字段（主兴趣、兴趣分布、证据文本等）
- 产物：`output/user_interest_profiles.json`

### 步骤④：一键聚类 + 画像（`cluster_and_infer_text2vec`）
- 入口：Web 调接口 `POST /api/cluster_and_infer_text2vec`
- 执行：由 `app_web/app.py` 串联触发 `train_tweet_topic` 和 `infer`
- 产物：完整的 `output/ml_artifacts/*` + 新的 `output/user_interest_profiles.json`

### 步骤⑤：簇深度命名（DeepSeek）
- 入口：Web 调接口 `POST /api/tweet_cluster_llm/run`
- 执行：`app_pipeline/cluster_llm_labels.py`（由 `app_web/app.py` 后台线程触发）
- 做的事：从各簇样本文本抽取内容，调用 DeepSeek 生成中文四字标签 + 人群画像/共性感知/总结
- 产物：
  - `output/ml_artifacts/cluster_label_map.json`
  - `output/ml_artifacts/cluster_llm_topic.json`

## 3.1) 训练与推断一句话定义（补充）

- `train_tweet_topic` 负责“训练模型”，`infer` 负责“用模型生成用户画像”。

### 训练阶段脚本（`train_tweet_topic.py`）

作用是把微博文本训练成“可用于画像的模型产物”。  
主要做：
- 微博文本向量化（text2vec）
- PCA 降维 + KMeans 聚类（找话题簇、搜索较优 `k`）
- 基于簇结果构造用户多标签目标
- 训练 GBDT 多标签分类器

## 4) 核心技术重点：轻量级 BERT + GBDT

### 4.1 轻量级 BERT（text2vec / sentence-transformers）
- 位置：`app_pipeline/train_tweet_topic.py`、`app_pipeline/infer.py`
- 作用：将中文微博文本转成语义向量，用于 PCA、KMeans、向量聚合和推断解释
- 价值：轻量高效、易部署、可复用，不走重型端到端微调

### 4.2 GBDT 多标签分类
- 训练：`app_pipeline/train_tweet_topic.py`（`TweetMultilabelGBDT`，内部使用 `GradientBoostingClassifier`）
- 推断：`app_pipeline/infer.py`（加载 `gbdt_multilabel.pkl` 输出标签概率与分类结果）
- 价值：对中等规模业务数据稳定、可控，且便于与规则和解释层结合

## 5) Web 端如何呈现结果

- `/profiles`：用户画像总览（昵称优先，缺失回退 UID）
- `/users/{user_id}`：用户兴趣详情（推断摘要 + 话题解释）
- `/users/{user_id}/weibo-sources`：原始微博溯源（回看具体证据文本）
- 任务运行机制：前端按钮提交任务 -> 后端创建 `job_id` -> 后台线程执行 -> 前端轮询 `GET /api/jobs/{job_id}` 展示进度与日志

## 6) 总结

这是一个可运行、可迭代、可追溯、可解释的微博兴趣画像平台。  
它不是单点模型脚本，而是完整业务流水线：**抓取 -> 数据组织 -> 轻量级 BERT 向量化 -> KMeans 结构化 -> GBDT 多标签分类 -> 画像展示与溯源**。
