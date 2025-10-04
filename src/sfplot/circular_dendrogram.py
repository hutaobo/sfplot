import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform
from pycirclize import Circos


def plot_circular_dendrogram_pycirclize(
    matrix: pd.DataFrame | np.ndarray,
    output_pdf: str,
    metric: str = "euclidean",
    method: str = "average",
    figsize: tuple[float, float] = (12, 12),
    r_lim: tuple[float, float] = (20, 100),
    leaf_label_size: int = 6
):
    """
    用 pyCirclize 绘制无缺口圆形树状图（PDF 矢量图）。

    参数
    -------
    matrix : DataFrame 或 ndarray
        如果是方阵且对称，当作距离矩阵；否则当作特征矩阵。
    output_pdf : str
        输出文件路径，例如 "./figures/tree.pdf"
    metric : str
        pdist 距离度量（特征矩阵时才用到）。
    method : str
        linkage 聚类方法。
    figsize : (w, h)
        最终 PDF 画布大小（英寸）。
    r_lim : (inner, outer)
        树的内外半径（pyCirclize 百分制坐标）。
    leaf_label_size : int
        叶子标签字号。
    """
    # ---------- 1. 解析数据 ----------
    if isinstance(matrix, pd.DataFrame):
        data = matrix.values
        labels = list(matrix.index)
    else:
        data = np.asarray(matrix)
        labels = [str(i) for i in range(data.shape[0])]

    # ---------- 2. 生成距离并聚类 ----------
    if data.shape[0] == data.shape[1] and np.allclose(data, data.T, atol=1e-8):
        dist_mat = data
    else:
        dist_mat = squareform(pdist(data, metric=metric))
    Z = linkage(dist_mat, method=method)

    # ---------- 3. linkage → Newick ----------
    root, _ = to_tree(Z, rd=True)

    def to_newick(node, parent_dist):
        name = labels[node.id] if node.is_leaf() else ""
        length = parent_dist - node.dist
        if node.is_leaf():
            return f"{name}:{length:.6f}"
        return f"({to_newick(node.left, node.dist)},{to_newick(node.right, node.dist)}):{length:.6f}"

    newick_str = to_newick(root, root.dist) + ";"
    tmp_nwk = output_pdf + ".nwk"
    with open(tmp_nwk, "w") as f:
        f.write(newick_str)

    # ---------- 4. 用 pyCirclize 绘图 ----------
    circos, tv = Circos.initialize_from_tree(  # 没有 figsize 参数
        tmp_nwk,
        r_lim=r_lim,
        leaf_label_size=leaf_label_size,
        line_kws=dict(color="black", lw=1.0),
        label_formatter=lambda t: rf"$\it{{{t}}}$"
    )

    fig = circos.plotfig()  # 返回 Matplotlib Figure
    fig.set_size_inches(figsize)  # 这里再调整图像尺寸
    fig.savefig(output_pdf, bbox_inches="tight")
    os.remove(tmp_nwk)
    print(f"✓ 圆形树状图已保存：{output_pdf}")


# plot_circular_dendrogram_pycirclize(
#     matrix=row_coph,
#     output_pdf="./figures/circular.pdf",
#     metric='euclidean',
#     method='average',
#     leaf_label_size=7,
#     figsize=(13, 13)      # 需要更大就改这里
# )
