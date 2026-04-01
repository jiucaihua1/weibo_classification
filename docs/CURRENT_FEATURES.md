# 当前功能汇总

## 1. 爬虫能力（保留原始能力）

- 用户微博采集：`python weibospider/run_spider.py tweet_by_user_id`
- 微博评论采集：`python weibospider/run_spider.py comment`
- 其他模式仍保留：`user` / `fan` / `follow` / `repost` / `tweet_by_tweet_id` / `tweet_by_keyword`

## 2. 数据输出能力

- 原始输出：`output/{spider_name}_*.jsonl`
- 统一输出：`output/unified_*.jsonl`
  - 核心字段：`user_id` / `text` / `source_type` / `created_at`
- 按用户聚合：`output/user_aggregate_*.json`

## 3. MVP 建模能力（app_pipeline）

- 文本预处理：`app_pipeline/preprocess.py`
  - URL/@/话题/表情清洗 + 中文分词
- 训练：`app_pipeline/train.py`
  - TF-IDF + KMeans + 固定兴趣类别映射
- 推断：`app_pipeline/infer.py`
  - 输出 `output/user_interest_profiles.json`
- 指标：`output/ml_artifacts/metrics.json`
  - `silhouette_score` / `label_coverage` / `label_distribution`

## 4. Web 展示能力

- FastAPI + Jinja 页面：`app_web/app.py`
- 页面：
  - `/`：用户画像列表 + 类别统计
  - `/users/{user_id}`：用户画像详情
- API：
  - `/api/profiles`
  - `/api/profiles/{user_id}`
  - `/api/stats`

## 5. 运行方式

- 一键启动：`python run_web.py`
- 页面端完成采集、训练、推断、查看

## 6. 当前已知限制

- 评论相关速度受微博接口 403 限流影响明显
- “给 user_id 直接全网抓该用户评论历史”不支持，只能用样本微博筛选法近似
