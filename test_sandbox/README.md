## test_sandbox

这里放一些**临时测试脚本**，不影响主流程。

### 可视化聚类分布（PCA + t-SNE + KMeans）

默认按**双数据集合并**后的向量（需先跑完下面「数据流水线」）：

- `output/text2vec_merged/user_features_text2vec.npy`
- `output/text2vec_merged/user_ids_text2vec.pkl`

### 多数据集流水线（自动合并 `output/unified_*.jsonl`）

```powershell
# 1) 合并：不传参数时自动扫描 output 下所有 unified_*.jsonl（按文件名排序）
.\.venv\Scripts\python.exe .\concat_user_texts.py
# 或按修改时间顺序： --sort-unified mtime
# 或指定目录： --output-dir output
# 或手动指定文件： .\concat_user_texts.py --input path\a.jsonl --input path\b.jsonl

# 2) 清洗
.\.venv\Scripts\python.exe -m app_pipeline.clean_concat_user_texts

# 3) text2vec 向量
.\.venv\Scripts\python.exe -m app_pipeline.step1_bert_features --cleaned-input "output/cleaned_user_texts_merged.jsonl" --outdir "output/text2vec_merged" --top-k 10 --device cpu
```

运行：

```powershell
.\.venv\Scripts\python.exe .\test_sandbox\cluster_viz.py --clusters 5
```

脚本还会在聚类完成后，读取 `output/cleaned_user_texts_merged.jsonl`，按簇做 **jieba + TF-IDF** 打印每类 Top 关键词。可改路径或关闭：

```powershell
.\.venv\Scripts\python.exe .\test_sandbox\cluster_viz.py --clusters 5 --texts-jsonl "output/cleaned_user_texts_merged.jsonl"
.\.venv\Scripts\python.exe .\test_sandbox\cluster_viz.py --skip-keywords
```

输出图片：

- `output/viz_cluster_tsne.png`

