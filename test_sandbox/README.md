## test_sandbox

这里放一些**临时测试脚本**，不影响主流程。

### 可视化聚类分布（PCA + t-SNE + KMeans）

默认读取你刚生成的向量特征：

- `output/text2vec_20260402224047/user_features_text2vec.npy`
- `output/text2vec_20260402224047/user_ids_text2vec.pkl`

运行：

```powershell
.\.venv\Scripts\python.exe .\test_sandbox\cluster_viz.py --clusters 5
```

输出图片：

- `output/viz_cluster_tsne.png`

