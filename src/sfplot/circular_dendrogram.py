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
    Draw a gapless circular dendrogram (PDF vector image) using pyCirclize.

    Parameters
    -------
    matrix : DataFrame or ndarray
        Treated as a distance matrix if square and symmetric; otherwise as a feature matrix.
    output_pdf : str
        Output file path, e.g. "./figures/tree.pdf"
    metric : str
        pdist distance metric (used only for feature matrices).
    method : str
        Linkage clustering method.
    figsize : (w, h)
        Final PDF canvas size in inches.
    r_lim : (inner, outer)
        Inner and outer radii of the tree (pyCirclize percent coordinates).
    leaf_label_size : int
        Font size for leaf labels.
    """
    # ---------- 1. Parse data ----------
    if isinstance(matrix, pd.DataFrame):
        data = matrix.values
        labels = list(matrix.index)
    else:
        data = np.asarray(matrix)
        labels = [str(i) for i in range(data.shape[0])]

    # ---------- 2. Generate distances and cluster ----------
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

    # ---------- 4. Draw with pyCirclize ----------
    circos, tv = Circos.initialize_from_tree(  # no figsize parameter
        tmp_nwk,
        r_lim=r_lim,
        leaf_label_size=leaf_label_size,
        line_kws=dict(color="black", lw=1.0),
        label_formatter=lambda t: rf"$\it{{{t}}}$"
    )

    fig = circos.plotfig()  # returns a Matplotlib Figure
    fig.set_size_inches(figsize)  # adjust image size here
    fig.savefig(output_pdf, bbox_inches="tight")
    os.remove(tmp_nwk)
    print(f"✓ Circular dendrogram saved: {output_pdf}")


# plot_circular_dendrogram_pycirclize(
#     matrix=row_coph,
#     output_pdf="./figures/circular.pdf",
#     metric='euclidean',
#     method='average',
#     leaf_label_size=7,
#     figsize=(13, 13)      # increase this if a larger figure is needed
# )
