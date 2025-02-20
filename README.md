## 安装

```python
pip install git+https://github.com/hutaobo/sfplot.git
pip install git+https://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@github.com/hutaobo/sfplot.git
```

## 功能

### 生成 Cluster Distance Heatmap

#### 使用 `base_path` 和 `sample` 加载数据并生成热图

```python
from sfplot import generate_cluster_distance_heatmap_from_path

base_path = "/path/to/data"
sample = "sample_name"
output_dir = "/path/to/output"

generate_cluster_distance_heatmap_from_path(base_path, sample, output_dir)
```

#### 直接使用 AnnData 对象生成热图
```python
import anndata
from sfplot import generate_cluster_distance_heatmap_from_adata, load_xenium_data

# 加载数据
base_path = "/path/to/data"
sample = "sample_name"
folder = os.path.join(base_path, sample)
adata = load_xenium_data(folder)
adata.uns["sample"] = sample  # 用于设置热图标题

# 生成热图
generate_cluster_distance_heatmap_adata(
    adata=adata,
    cluster_col="Cluster",
    output_dir="/path/to/output"
)
```
