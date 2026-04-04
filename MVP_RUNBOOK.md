# 微博兴趣分类 MVP 运行说明

## 1) 环境安装

```bash
pip install -r requirements.txt
```

## 2) 采集输出（统一 schema）

在 `weibospider` 目录执行采集后，`output/` 会生成：

- `unified_*.jsonl`：统一结构事件流，字段包含 `user_id/text/source_type/created_at`
- `user_aggregate_*.json`：按 `user_id` 聚合后的记录

示例命令：

```bash
cd weibospider
python run_spider.py tweet_by_user_id
python run_spider.py comment
```

### 推荐：三步多维采集（关键词召回 -> 用户资料 -> 历史推文）

为了减少改动核心爬虫代码、同时提升样本覆盖与均衡度，可直接使用根目录脚本：

```bash
python multidim_crawl.py \
  --keywords "手机评测,骁龙,AIGC,ETF,原神" \
  --max-pages 5 \
  --per-keyword-limit 80 \
  --total-limit 600 \
  --append-user-ids
```

为降低微博搜索 418 风控，脚本默认设置 `WEIBO_SPLIT_BY_HOUR=false`。如需指定时间范围可追加：

```bash
python multidim_crawl.py \
  --keywords "AIGC,手机评测" \
  --max-pages 2 \
  --start-time "2026-03-01 00:00:00" \
  --end-time "2026-04-01 23:59:59"
```

该脚本会自动执行：

- 第一步：`tweet_by_keyword` 召回候选人群；
- 第二步：从 `output/unified_*.jsonl` 提取并均衡抽样 `user_id`，写入 `input/user_ids.txt`；
- 第三步：依次执行 `user` 与 `tweet_by_user_id`，得到静态资料 + 历史内容的多维数据。

## 3) 一键启动 Web 系统

```bash
python run_web.py
```

启动后在页面填写 `user_id` 与 `cookie`，点击“开始运行”，系统会自动执行：

- `user` 信息采集
- `tweet_by_user_id` 采集
- `app_pipeline.train_tweet_topic` 训练
- `app_pipeline.infer` 推断

## 4) Web 页面

- 首页：`http://127.0.0.1:8000/`
- 接口：
  - `GET /api/profiles`
  - `GET /api/profiles/{user_id}`
  - `GET /api/stats`

## 5) 最小评估说明

训练后会在 `output/ml_artifacts/metrics.json` 生成最小评估结果：

- `silhouette_score`：聚类可分性指标（越高越好）
- `label_coverage`：8 个固定兴趣类覆盖率
- `label_distribution`：预测类别分布

建议再做 20-50 条人工抽样核验（检查 Top1 兴趣是否符合用户历史内容），用于快速验证业务可用性。
