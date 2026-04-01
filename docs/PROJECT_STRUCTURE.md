# 项目分层结构
[README.md](../README.md)
## 核心目录

- `weibospider/`：原始爬虫工程（采集层）
- `app_pipeline/`：训练与推断模块
  - `preprocess.py`：文本清洗、分词
  - `train.py`：TF-IDF + KMeans 训练
  - `infer.py`：用户兴趣推断
- `app_web/`：FastAPI 与 Jinja 页面
  - `app.py`：Web API 与任务编排
  - `templates/`：页面模板
- `run_web.py`：统一启动入口（推荐）
- `output/`：采集数据、模型产物、画像结果
- `docs/`：补充文档
