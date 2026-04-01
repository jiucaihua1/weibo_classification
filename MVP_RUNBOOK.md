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

## 3) 一键启动 Web 系统

```bash
python run_web.py
```

启动后在页面填写 `user_id` 与 `cookie`，点击“开始运行”，系统会自动执行：

- `user` 信息采集
- `tweet_by_user_id` 采集
- `app_pipeline.train` 训练
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
