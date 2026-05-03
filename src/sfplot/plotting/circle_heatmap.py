import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def circle_heatmap(bg_df: pd.DataFrame,
                   circle_df: pd.DataFrame,
                   *,
                   cmap: str = "RdBu",
                   size_exponent: float = 1.0,
                   circle_fill: str = "white",
                   circle_edge: str = "black",
                   circle_edge_lw: float = 0.5,
                   add_legend: bool = True,
                   legend_title: str = "Transcript Percentage (%)",
                   figsize: tuple = (8, 6),
                   ax: plt.Axes = None):
    """
    绘制结合热图和圆圈的图：
      - bg_df: 0–1 之间的分值，用红-白-蓝表示；
      - circle_df: 0–100 (%) 的百分比，用圆圈面积体现；
      - 0% 不画圆，100% 精确映射成单元格直径的圆；
      - 图例只展示 [5,25,45,65,85] 这五个百分比。
    """
    # 1. 校验
    if bg_df.shape != circle_df.shape:
        raise ValueError("bg_df 和 circle_df 必须同形状")
    if not (bg_df.index.equals(circle_df.index) and bg_df.columns.equals(circle_df.columns)):
        raise ValueError("bg_df 和 circle_df 必须同索引和列")

    # 2. 创建 Figure/Axes
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])
        if add_legend:
            heatmap_legend_ax = fig.add_axes([0.8, 0.55, 0.15, 0.1])
            circle_legend_ax  = fig.add_axes([0.8, 0.15, 0.15, 0.15])
        else:
            heatmap_legend_ax = circle_legend_ax = None
    else:
        fig = ax.figure
        if add_legend:
            heatmap_legend_ax = fig.add_axes([0.8, 0.55, 0.15, 0.1])
            circle_legend_ax  = fig.add_axes([0.8, 0.15, 0.15, 0.15])
        else:
            heatmap_legend_ax = circle_legend_ax = None

    nrows, ncols = bg_df.shape

    # 3. 绘背景热图
    ax.pcolormesh(bg_df.values, cmap=cmap, edgecolors='white', linewidths=0.1)
    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(bg_df.columns, rotation=90, ha="center", fontsize=12)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(bg_df.index,
                       fontdict={'fontstyle': 'italic',
                                 'fontname': 'Arial',
                                 'fontsize': 12})
    ax.invert_yaxis()

    # 4. 计算单元格大小 & 最大圆面积
    fig.canvas.draw()
    w_px = ax.bbox.width / ncols
    w_pt = w_px * 72 / fig.dpi
    max_area = np.pi * (w_pt / 2) ** 2

    # 5. 映射 circle_df → sizes（0%→0；100%→max_area）
    vals = circle_df.values.astype(float)
    frac = np.clip(vals / 100.0, 0, 1)
    sizes = (frac ** size_exponent) * max_area
    sizes[vals == 0] = 0

    # 6. 画圆（仅 size>0）
    xg, yg = np.meshgrid(np.arange(ncols) + 0.5,
                         np.arange(nrows) + 0.5)
    xs, ys, ss = xg.ravel(), yg.ravel(), sizes.ravel()
    mask = ss > 0
    ax.scatter(xs[mask], ys[mask],
               s=ss[mask],
               facecolors=circle_fill,
               edgecolors=circle_edge,
               linewidths=circle_edge_lw,
               zorder=10)

    # 7. 热图图例
    if add_legend and heatmap_legend_ax is not None:
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        heatmap_legend_ax.imshow(grad, aspect='auto', cmap=cmap)
        heatmap_legend_ax.set_xticks([0, 255])
        heatmap_legend_ax.set_xticklabels(["0", "1"], fontsize="small")
        heatmap_legend_ax.set_yticks([])
        heatmap_legend_ax.set_title("Spatial Separation Score",
                                    fontsize="small")

    # 8. 圆圈大小图例 — 只显示 [5,25,45,65,85]
    if add_legend and circle_legend_ax is not None:
        circle_legend_ax.clear()
        circle_legend_ax.axis('off')
        circle_legend_ax.set_title(legend_title, fontsize="small")

        legend_vals = np.array([5, 25, 45, 65, 85], dtype=float)
        legend_frac = legend_vals / 100.0
        legend_sizes = (legend_frac ** size_exponent) * max_area

        x_leg = np.arange(len(legend_vals)) * 4.0
        y_leg = np.full(len(legend_vals), 0.5)
        for i, (v, sz) in enumerate(zip(legend_vals, legend_sizes)):
            circle_legend_ax.scatter(
                x_leg[i], y_leg[i],
                s=sz,
                facecolors=circle_fill,
                edgecolors=circle_edge,
                linewidths=circle_edge_lw,
                zorder=10
            )
            circle_legend_ax.text(
                x_leg[i], y_leg[i] - 0.3,
                f"{int(v)}",
                ha="center", va="top", fontsize="small"
            )
        circle_legend_ax.set_xlim(-1, x_leg[-1] + 1)
        circle_legend_ax.set_ylim(0, 1.5)

    return fig, ax, {"heatmap": heatmap_legend_ax,
                     "circle": circle_legend_ax}


# ==== 示例 ====
if __name__ == "__main__":
    genes = ['Gsdma3','Kcna10','Ly6k','Dsg1a','Pou4f3']
    cells = [str(i) for i in range(1, 30)]
    np.random.seed(0)
    bg = np.random.rand(len(genes), len(cells))
    circ = np.random.rand(len(genes), len(cells)) * 100
    circ[0,0] = 0
    circ[0,1] = 100

    bg_df = pd.DataFrame(bg, index=genes, columns=cells)
    circ_df = pd.DataFrame(circ, index=genes, columns=cells)

    fig, ax, legends = circle_heatmap(bg_df, circ_df)
    plt.show()
