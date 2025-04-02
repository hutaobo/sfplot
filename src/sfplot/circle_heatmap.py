import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def circle_heatmap(bg_df: pd.DataFrame,
                   circle_df: pd.DataFrame,
                   *,
                   cmap: str = "RdBu_r",
                   smallest_circle: float = 20,
                   largest_circle: float = 200,
                   size_exponent: float = 1.0,
                   circle_fill: str = "white",
                   circle_edge: str = "black",
                   circle_edge_lw: float = 0.5,
                   add_legend: bool = True,
                   legend_title: str = "Percentage (%)",
                   figsize: tuple = (8, 6),
                   ax: plt.Axes = None):
    """
    绘制一个结合热图和圆圈的图：
      - 背景使用 `bg_df` 中的数值（范围 0-1）绘制热图，使用红-白-蓝渐变，
        其中 0 显示为最红，1 显示为最蓝；
      - 每个单元格中央绘制一个圆圈，其大小根据 `circle_df` 中的百分比数据决定，
        该数据自动计算最小值和最大值；圆圈填充为白色，边框为黑色且边框较细；
      - 可在图右侧添加圆圈大小的标尺。

    参数
    ----
    bg_df : pandas.DataFrame
        背景热图数据，每行代表一个基因、每列代表一种细胞类型，数值应在 0 到 1 范围内。
    circle_df : pandas.DataFrame
        圆圈大小数据，行列与 bg_df 对应，数值通常为百分比（例如 0-100）。
    cmap : str, 默认 "RdBu_r"
        热图背景的颜色映射，使用红-白-蓝渐变（0 为最红，1 为最蓝）。
    smallest_circle : float, 默认 20
        最小圆圈的面积（scatter 中 s 参数，单位为点²）。
    largest_circle : float, 默认 200
        最大圆圈的面积。
    size_exponent : float, 默认 1.0
        用于调整圆圈大小的指数（控制大小分布）。
    circle_fill : str, 默认 "white"
        圆圈填充颜色。
    circle_edge : str, 默认 "black"
        圆圈边框颜色。
    circle_edge_lw : float, 默认 0.5
        圆圈边框线宽。
    add_legend : bool, 默认 True
        是否在图右侧添加圆圈大小的标尺。
    legend_title : str, 默认 "Percentage (%)"
        标尺的标题。
    figsize : tuple, 默认 (8,6)
        图形尺寸。
    ax : matplotlib.axes.Axes, 可选
        如果提供，则在该轴上绘图；否则新建一个图形和轴。

    返回
    -------
    fig, ax, legend_ax
        分别为主图的 Figure、Axes，以及标尺的 Axes（如果 add_legend 为 False，则 legend_ax 为 None）。
    """
    # 检查两个 DataFrame 形状及索引是否一致
    if bg_df.shape != circle_df.shape:
        raise ValueError("bg_df 和 circle_df 必须具有相同的形状")
    if not (bg_df.index.equals(circle_df.index) and bg_df.columns.equals(circle_df.columns)):
        raise ValueError("bg_df 和 circle_df 必须具有相同的索引和列")

    # 创建图形与轴：若未提供 ax，则使用 GridSpec 分成主图和标尺两个区域
    if ax is None:
        fig = plt.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.3)
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
    else:
        fig = ax.figure
        legend_ax = None
        if add_legend:
            # 如果提供了 ax 但需要标尺，则在图中额外创建一个轴（位置可根据需要调整）
            legend_ax = fig.add_axes([0.85, 0.15, 0.1, 0.7])

    # 绘制热图背景
    # 使用 pcolor 绘制，每个格子的左上角坐标为 (i, j)
    heat = ax.pcolor(bg_df.values, cmap=cmap, edgecolors='none')

    # 设置轴刻度和标签
    nrows, ncols = bg_df.shape
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(bg_df.columns, rotation=90, ha="center", fontsize="small")
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(bg_df.index, fontsize="small")
    ax.invert_yaxis()  # 使第一行在上

    # 处理圆圈大小数据（自动计算范围）
    circ_values = circle_df.values.astype(float)
    min_val = np.nanmin(circ_values)
    max_val = np.nanmax(circ_values)
    if max_val == min_val:
        frac = np.ones_like(circ_values)
    else:
        frac = (circ_values - min_val) / (max_val - min_val)
    # 根据比例计算圆圈面积
    sizes = (frac ** size_exponent) * (largest_circle - smallest_circle) + smallest_circle

    # 构造网格坐标（每个单元格中心位置）
    x_coords, y_coords = np.meshgrid(np.arange(ncols) + 0.5, np.arange(nrows) + 0.5)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    sizes_flat = sizes.flatten()

    # 绘制圆圈，scatter 的 s 参数表示面积（点²）
    ax.scatter(
        x_coords, y_coords,
        s=sizes_flat,
        facecolors=circle_fill,
        edgecolors=circle_edge,
        linewidths=circle_edge_lw,
        zorder=10
    )

    # 如果需要，添加圆圈大小标尺
    if add_legend and legend_ax is not None:
        legend_ax.clear()
        legend_ax.axis('off')
        legend_ax.set_title(legend_title, fontsize="small")

        # 生成一组代表性数值（例如 5 个，从 max 到 min）
        n_steps = 5
        legend_vals = np.linspace(max_val, min_val, n_steps)
        if max_val == min_val:
            legend_frac = np.ones(n_steps)
        else:
            legend_frac = (legend_vals - min_val) / (max_val - min_val)
        legend_sizes = (legend_frac ** size_exponent) * (largest_circle - smallest_circle) + smallest_circle

        # 绘制标尺散点：横向排列
        x_leg = np.arange(n_steps) * 1.5
        y_leg = np.full(n_steps, 0.5)
        legend_ax.scatter(
            x_leg, y_leg,
            s=legend_sizes,
            facecolors=circle_fill,
            edgecolors=circle_edge,
            linewidths=circle_edge_lw,
            zorder=10
        )
        # 添加对应数值标签
        for i, val in enumerate(legend_vals):
            legend_ax.text(
                x_leg[i], y_leg[i] - 0.3,
                f"{val:.1f}",
                ha="center", va="top", fontsize="small"
            )
        legend_ax.set_xlim(-1, x_leg[-1] + 1)
        legend_ax.set_ylim(0, 1.5)

    # 返回图形、主轴和标尺轴
    return fig, ax, legend_ax

# 示例用法
if __name__ == "__main__":
    # 假设你已经有两个 DataFrame，下面用随机数据作示例
    genes = [f"Gene{i}" for i in range(1, 11)]
    cell_types = ["B cell", "CD4+ T cell", "CD8+ T cell", "Endothelial cell", "Epithelial cell"]
    np.random.seed(0)
    # bg_df 中的值在 0-1 之间
    bg_data = np.random.rand(len(genes), len(cell_types))
    bg_df = pd.DataFrame(bg_data, index=genes, columns=cell_types)
    # circle_df 中的值在 0-100 之间（模拟百分比）
    circle_data = np.random.rand(len(genes), len(cell_types)) * 100
    circle_df = pd.DataFrame(circle_data, index=genes, columns=cell_types)

    fig, ax, legend_ax = circle_heatmap(bg_df, circle_df)
    plt.show()
