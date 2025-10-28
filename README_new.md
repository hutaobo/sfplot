SFplot 使用说明

简介

sfplot 是一个用于绘制“Search-and-Find Plot (SFplot)”的 Python 工具包，主要针对空间转录组等数据进行分析和可视化。它能够计算细胞类型簇之间的空间距离矩阵（基于 cophenetic 距离）并生成聚类热图（结构图/距离图），帮助研究者理解不同细胞群在空间上的分布关系。除了核心的距离计算和热图绘制函数，sfplot 还提供加载 10x Xenium 数据、串行/并行分析流程以及GUI界面等实用功能。

以下按照模块对本仓库的函数进行整理，每个函数包含作用说明、主要参数和返回值，以及示例用法。如果示例代码需要依赖其它函数，会同时注明依赖关系或调用顺序。

模块: data_processing.py

函数: load_xenium_data(folder: str, normalize: bool = True) -> anndata.AnnData

作用: 加载并预处理指定文件夹下的 10x Xenium 数据，返回包含表达矩阵及注释信息的 AnnData 对象。函数内部会自动寻找 Xenium 输出中的聚类结果和 UMAP 坐标信息，并将其整合到 AnnData 中(例如在 adata.obs["Cluster"] 和 adata.obsm["X_umap"] 中)。如果数据未解压，函数会尝试解压 analysis.tar.gz 或直接读取 analysis.h5 以获取聚类和降维信息。处理过程中也可以选择对数据进行归一化和对数转换。

参数:
- folder: Xenium 数据所在目录路径, 需包含 analysis 子文件夹或相关压缩文件。
- normalize: 是否对数据执行标准的归一化和 log1p 转换, 以及尺度缩放(默认为 True)。

返回: 返回一个 AnnData 对象, 包含加载的细胞信息 (adata.obs 含 cell_id、Cluster 等) 和表达矩阵 (adata.X 等)。若 normalize=True, 则数据已归一化并存入 adata.raw 中。

示例:
from sfplot import load_xenium_data
adata = load_xenium_data("/path/to/xenium_folder", normalize=True)
print(adata.obs[["cell_id", "Cluster"]].head())  # 查看前几条细胞的聚类标签

示例依赖: 无特殊依赖, 但要求 folder 路径下存在 Xenium 的标准输出文件结构。

模块: circular_dendrogram.py

函数: plot_circular_dendrogram_pycirclize(matrix, output_pdf: str, metric: str = "euclidean", method: str = "average", figsize: tuple = (12, 12), r_lim: tuple = (20, 100), leaf_label_size: int = 6) -> None

作用: 使用 pyCirclize 绘制圆形树状图 (dendrogram) 并保存为 PDF 文件。给定一个距离矩阵或特征矩阵, 函数先进行层次聚类, 生成 Newick 树格式, 然后利用 pyCirclize 将树状图绘制成圆形、不闭合缺口的形式。这对于直观展示簇间关系非常有用。

参数:
- matrix: 输入的数据, 可以是方阵距离矩阵(numpy.ndarray 或 pandas.DataFrame, 且对称), 或特征矩阵(函数内部会计算其欧氏距离)。
- output_pdf: 输出 PDF 文件路径, 例如 "./figures/tree.pdf", 函数会将绘制结果保存至该文件。
- metric: 计算距离所用的度量方式, 当提供特征矩阵时使用(默认 "euclidean")。
- method: 层次聚类使用的 linkage 方法(默认 "average")。
- figsize: 生成图像的尺寸(宽, 高), 单位为英寸。
- r_lim: 圆形树状图的内、外半径设置(pyCirclize 使用0-100的百分比坐标)。
- leaf_label_size: 叶节点标签的字体大小。

返回: 无返回值。函数执行完毕后, 会在指定路径生成一个 PDF 文件的圆形树状图, 并在控制台打印保存成功的信息。

示例:
from sfplot import plot_circular_dendrogram_pycirclize
# 假设 row_coph 是一个 DataFrame(簇间距离矩阵, 例如 compute_cophenetic_distances_from_df 的输出)
plot_circular_dendrogram_pycirclize(matrix=row_coph, output_pdf="./circular_tree.pdf",
                                    metric="euclidean", method="average", leaf_label_size=7)
# 输出会生成 circular_tree.pdf, 其中绘制了 row_coph 距离矩阵的圆形树状图

示例依赖: 需要先计算得到距离矩阵 row_coph, 可由 compute_cophenetic_distances_from_df 等函数获得。

模块: ghost_Searcher_with_Findee.py

函数: compute_groupwise_average_distance_between_two_dfs(df1: pd.DataFrame, df2: pd.DataFrame, df1_x_col: str = 'x', df1_y_col: str = 'y', df1_celltype_col: str = 'celltype', df2_x_col: str = 'x', df2_y_col: str = 'y', df2_celltype_col: str = 'celltype', n_jobs: int = -1) -> pd.DataFrame

作用: 计算两个数据集之间分组的平均最近邻距离矩阵。具体来说, 对于数据集1(df1)中的每一种“源”类别(如细胞类型), 计算其所有点到数据集2(df2)中每一种“目标”类别的所有点的最近邻距离的平均值。结果以 DataFrame 矩阵返回, 行索引是 df1 中的各源类别, 列索引是 df2 中的各目标类别, 元素为平均最近邻距离。该函数通常用于比较两类实体在空间上的邻近程度, 例如比较细胞簇和转录本簇的距离。

参数:
- df1, df2: 两个输入的 DataFrame, 应分别包含 X, Y 坐标以及类别标识列。例如 df1 可以是细胞数据, df2 可以是转录本数据。
- df1_x_col, df1_y_col: df1 中表示坐标的列名(默认为 'x' 和 'y')。
- df1_celltype_col: df1 中表示细胞类型的列名(默认 'celltype')。
- df2_x_col, df2_y_col, df2_celltype_col: 对应 df2 的列名, 含义同上。
- n_jobs: 并行计算使用的进程数(默认为 -1 表示利用所有 CPU 线程)。

返回: 返回一个 pandas DataFrame, 形状为 (df1_unique_types × df2_unique_types)。行索引为 df1 中 unique 的源类别, 列索引为 df2 中 unique 的目标类别。每个单元格的值是对应源类别和目标类别之间平均最近邻距离。若某源类别或目标类别不存在数据点, 将填入 NaN。

示例:
from sfplot import compute_groupwise_average_distance_between_two_dfs
# df_cells: 包含细胞坐标和类型的 DataFrame(列包括 'x','y','celltype')
# df_transcripts: 包含转录本坐标和类型的 DataFrame(列包括 'x','y','celltype')
avg_dist_df = compute_groupwise_average_distance_between_two_dfs(df_cells, df_transcripts)
print(avg_dist_df.head())  # 查看细胞类型 vs 转录本类型 的平均距离矩阵

示例依赖: 需要准备好两个 DataFrame, 并确保列名匹配函数参数要求。

模块: Searcher_Findee_Score.py

该模块实现了 SFplot 的核心算法, 包括计算 cophenetic 距离矩阵和绘制热图等函数。

函数: compute_cophenetic_distances_from_adata(adata: anndata.AnnData, cluster_col: str = "Cluster", output_dir: Optional[str] = None, method: str = "average") -> Tuple[pd.DataFrame, pd.DataFrame]

作用: 从 AnnData 对象计算 cophenetic 距离矩阵(行、列两个方向), 并对结果分别归一化到 [0,1]。其中行矩阵表示细胞簇之间的距离(常称为 StructureMap), 列矩阵在某些情形下可代表另一维度(例如 Findee D Score)。函数假定 AnnData 的 .obs 中存在表示簇的列(默认名为 "Cluster"), 以及 .obsm["spatial"] 包含细胞坐标信息。

参数:
- adata: AnnData 对象, 要求 adata.obs 有 cluster_col 列, 以及 adata.obsm["spatial"] 提供空间坐标 (形状 n_cells × 2 或 3)。同时要求 adata.obs["cell_id"] 列用于标识细胞。
- cluster_col: obs 中表示细胞簇/类别的列名, 默认 "Cluster"。
- output_dir: 可选, 若提供则输出结果(如距离矩阵)会保存文件到此目录。默认 None 表示不保存, 只返回结果。
- method: 层次聚类的方法(linkage method), 默认为 "average" 平均链。

返回: 一个二元组 (row_cophenetic_df, col_cophenetic_df), 均为 pandas DataFrame。row_cophenetic_df 为按细胞簇聚类计算的 cophenetic 距离矩阵(行聚类方向), col_cophenetic_df 为列方向的 cophenetic 距离矩阵。矩阵的行列索引为簇的类别名, 且已分别归一化到 [0, 1]。(注意: 两矩阵的最小值和最大值是分别归一化的, 行、列各自独立)。

示例:
from sfplot import compute_cophenetic_distances_from_adata
row_coph, col_coph = compute_cophenetic_distances_from_adata(adata, cluster_col="Cluster", method="average")
print(f"Row matrix shape: {row_coph.shape}, Col matrix shape: {col_coph.shape}")
# 如需将结果保存至文件夹:
compute_cophenetic_distances_from_adata(adata, output_dir="./output")

示例依赖: adata 需预先通过 load_xenium_data 等获取, 并确保有需要的聚类和坐标信息。

函数: compute_cophenetic_distances_from_df(df: pd.DataFrame, x_col: str = "x", y_col: str = "y", z_col: Optional[str] = None, celltype_col: str = "celltype", output_dir: Optional[str] = None, method: str = "average", show_corr: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]

作用: 从给定的 DataFrame 计算 cophenetic 距离矩阵, 并将结果归一化到 [0,1] 后返回。与上述 AnnData 版本类似, 但直接使用 DataFrame 输入。DataFrame 应包含每个细胞/点的坐标(x_col, y_col, z_col)和类别标签(celltype_col)。该函数首先计算每个细胞相对于各簇(类别)的最近邻距离, 然后按簇取均值构建簇×簇的距离矩阵, 最后进行层次聚类并计算 cophenetic 距离。

参数:
- df: 输入的 pandas DataFrame, 需包含坐标和类别列。
- x_col, y_col, z_col: 坐标列名, z_col 可选(如果有第三维坐标)。默认使用二维 (x, y)。
- celltype_col: 类别列的名称, 即每个点所属簇的标识列, 默认 "celltype"。
- output_dir: 输出目录。如果提供, 将在计算过程中将一些结果文件保存到该目录(例如聚类结果 CSV 或调试信息); 默认为 None。
- method: 层次聚类方法, 默认 "average"。
- show_corr: 是否在计算后打印 cophenetic 相关系数, 默认 False 不打印。

返回: (row_cophenetic_df, col_cophenetic_df) 二元组, 与前述 AnnData 函数相似。行距矩阵 row_cophenetic_df 和列距矩阵 col_cophenetic_df 均为 DataFrame, 索引为簇名, 且各自归一化到 [0,1]。一般地, 在 Search-and-Find Plot 中, 我们主要关注行距矩阵(即结构距离矩阵); 列距矩阵在某些分析(如转录本作为类别)时会用到。

示例:
from sfplot import compute_cophenetic_distances_from_df
# 假设 df 包含列 'x','y','celltype'
row_coph, col_coph = compute_cophenetic_distances_from_df(df, x_col="x", y_col="y", celltype_col="celltype")
# 查看一个簇之间的距离:
print(row_coph.loc["ClusterA", "ClusterB"])

示例依赖: DataFrame 中必须包含所需的列, 且每个簇至少有一个点, 否则返回结果可能为空并抛出异常。

函数: plot_cophenetic_heatmap(matrix: pd.DataFrame, matrix_name: Optional[str] = None, output_dir: Optional[str] = None, output_filename: Optional[str] = None, figsize: Optional[tuple] = None, cmap: str = "RdBu", linewidths: float = 0.5, annot: bool = False, sample: str = "Sample", xlabel: Optional[str] = None, ylabel: Optional[str] = None, show_dendrogram: bool = True, quiet: bool = True, return_figure: bool = False, return_image: bool = False, dpi: int = 300) -> Optional[Union[seaborn.ClusterGrid, PIL.Image.Image]]

作用: 绘制给定距离矩阵的聚类热图(使用 seaborn.clustermap)并根据参数输出结果。此函数对绘图进行了多方面优化:
- 在保存 PDF 时保证文字可编辑(设置 Matplotlib 字体参数)。
- 自动调整图例(colorbar)和树状图(dendrogram)的布局, 使热图方格呈正方形, 树状图和图例位置美观。
- 可根据 matrix_name 自动设置标题和轴标签。例如, 传入 matrix_name="row_coph" 则标题会设为 "StructureMap of {sample}", X/Y轴标签分别为 "Searcher"。若为 "col_coph" 则标题为 "Findee's D score of {sample}"。
- 提供静默模式(quiet=True) 以屏蔽字体相关的警告日志。
- 可以选择返回 figure 对象或 PIL 图像, 用于GUI显示或进一步处理。

参数(摘录):
- matrix: 要绘制的距离矩阵 (pandas DataFrame), 通常为 compute_cophenetic_distances_from_df 等函数的输出之一。
- matrix_name: 可选, 用于识别矩阵类型的名称。例如 "row_coph" 或 "col_coph"; 若提供, 会影响默认标题和轴标签设定。
- output_dir & output_filename: 输出文件目录和文件名。如果不要求返回对象(即 return_figure 和 return_image 均为 False), 则函数将保存 PDF 文件到指定路径。文件名可自动根据 matrix_name 生成(如未提供则使用默认)。
- figsize: 图大小(宽,高), 若不提供则根据矩阵维度动态计算一个合适的大小。
- cmap: 热图颜色映射, 默认 "RdBu" 红蓝渐变。
- linewidths: 单元格边框宽度, 默认 0.5。
- annot: 是否在单元格中显示数值, 默认 False(不显示)。
- sample: 样本名称, 用于标题中或默认文件名中。
- xlabel, ylabel: 自定义横纵轴标签, 通常不需要手动设置(函数会根据 matrix_name 选择合适标签, 如 Searcher/Findee)。
- show_dendrogram: 是否显示行列的 dendrogram, 默认 True。如果设为 False, 将不进行聚类, 仅绘制原矩阵的热图。
- quiet: 是否静默某些日志输出(主要是字体相关警告), 默认 True 安静模式。
- return_figure, return_image: 控制函数返回值。如果 return_image=True 则返回 PIL.Image 对象(高分辨率图像), 否则如 return_figure=True 则返回 seaborn 的 ClusterGrid 对象。如果两者都 False, 则不返回, 将热图保存为文件。(注意: 当 return_image=True 时, 会忽略 return_figure 并优先返回图像)。
- dpi: 图像 DPI 分辨率, 仅当 return_image=True 时有效, 默认 300。

返回: 根据参数不同可能返回 None、ClusterGrid、或 PIL.Image。
- 如果 return_image=True: 返回绘制好的 PIL.Image.Image 对象(便于在 GUI 中直接显示或嵌入报告)。
- 否则如果 return_figure=True: 返回 seaborn.clustermap 的 ClusterGrid 对象(包含 figure 和 axes 信息)。
- 如果既不要求返回对象又提供了输出路径: 函数将把图保存为 PDF 并返回 None, 同时在控制台打印保存路径。

示例:
from sfplot import plot_cophenetic_heatmap
# 假设 row_coph 为 compute_cophenetic_distances_from_df 得到的 DataFrame (结构距离矩阵)
# 1) 将热图保存为 PDF 文件:
plot_cophenetic_heatmap(row_coph, matrix_name="row_coph", output_dir="./output", sample="MouseBrain")
# 2) 获取绘制结果为图像对象:
img = plot_cophenetic_heatmap(row_coph, matrix_name="row_coph", sample="MouseBrain", return_image=True)
img.save("temp.png")  # 将图像对象另存为文件 (也可直接用于 GUI 显示)

示例依赖: 需有准备好的距离矩阵 DataFrame。绘图本身依赖 matplotlib/seaborn 库, plot_cophenetic_heatmap 内部会调用辅助函数 _ensure_font() 来确保字体有效, 无需用户干预。

(附) 工具函数 silence(logger_name, level) 和 _ensure_font()

- silence: 一个上下文管理器(contextmanager), 用于暂时提高指定 logger 的日志级别, 从而在 with 块中静默某些日志输出。例如在绘图时静默 fontTools.subset 的调试信息。一般无需单独调用, 已经在 plot_cophenetic_heatmap 内部使用。
- _ensure_font: 内部函数, 确保绘图时使用系统中常见的 sans-serif 字体(优先 Arial, 否则降级为 Liberation Sans 或 DejaVu Sans), 避免 PDF 中中文或特殊字符缺失。该函数在绘图函数中自动调用, 不对外暴露。

模块: compute_cophenetic_distances_from_df_memory_opt.py

函数: pick_batch_size(n_cells: int, dims: int = 2, frac: float = 0.30, hard_min: int = 50000, hard_max: Optional[int] = None, bytes_per_row: Optional[int] = None, safety_gb: float = 8.0, env_override_var: str = "BATCH_SIZE_OVERRIDE") -> int

作用: 根据当前机器内存情况和数据规模, 计算合适的批处理大小。此函数用于内存优化场景, 比如在对大量细胞计算距离时, 通过分批处理降低内存占用。pick_batch_size 会尝试利用可用内存的 frac 比例, 同时保留一定的安全余量(safety_gb), 估算每个数据点处理所需内存(bytes_per_row), 最终得出一个介于 hard_min 和 hard_max 之间的批大小。函数也支持通过环境变量强制覆盖批大小。

参数:
- n_cells: 总的数据点数量(如细胞数), 用于计算批次数量上限。
- dims: 数据维度, 通常2或3; 维度越高, 每行处理内存需求可能增加(函数对 bytes_per_row 有默认调整)。
- frac: 打算使用的可用内存比例, 默认 0.30(30%)。函数会将机器可用内存减去安全余量后, 乘以该比例来预算可用内存。
- hard_min: 批大小下限, 不论内存多小, 至少处理这么多个数据点, 默认 50,000。
- hard_max: 批大小上限, 可选。None 表示不设上限, 否则不会超过此值。
- bytes_per_row: 每个数据点处理时消耗的字节数估计。如不提供, 函数会根据 dims 给一个保守默认值(dims<=2 时64字节, 否则80字节)。
- safety_gb: 保留的安全内存大小(GB), 默认 8.0, 即从可用内存中预留8GB不给批处理用, 以避免耗尽内存。
- env_override_var: 环境变量名, 如设置且有效, 则直接用该环境变量的值作为批大小, 忽略上述计算(用于快速实验)。

返回: 一个整数, 表示建议的批处理大小(不会小于 hard_min, 不会大于 hard_max 且不会超过 n_cells)。这个值可以用于控制循环或并行处理时每次处理的数据量, 防止内存占用过高。

示例:
from sfplot import pick_batch_size
batch = pick_batch_size(n_cells=200000, dims=3, frac=0.5, hard_min=10000, hard_max=50000)
print(f"Suggested batch size: {batch}")
# 如果需要, 可以通过设置环境变量 BATCH_SIZE_OVERRIDE 强制批大小:
os.environ["BATCH_SIZE_OVERRIDE"] = "30000"
batch2 = pick_batch_size(n_cells=200000)

示例依赖: 需确保安装了 psutil 库(函数内部使用它获取系统内存)。

函数: compute_cophenetic_distances_from_df_memory_opt(df: pd.DataFrame, x_col: str = "x", y_col: str = "y", z_col: Optional[str] = None, celltype_col: str = "celltype", method: str = "average", show_corr: bool = False, batch_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]

作用: 内存友好型的 DataFrame 版 cophenetic 距离计算。功能上与 compute_cophenetic_distances_from_df 相同, 但是通过分批处理最近邻计算来降低内存占用, 适合细胞数量极大的数据集。该函数会按簇逐列计算距离, 每次查询所有细胞到当前簇最近邻距离时, 如果总细胞数很大, 会根据给定或自动估计的 batch_size 分块计算, 避免一次性构建庞大的距离矩阵。

参数: 与 compute_cophenetic_distances_from_df 基本相同, 额外增加:
- batch_size: 批处理大小。如果提供, 则按照此批大小对细胞进行分批计算距离; 如果为 None, 函数内部会将一次性处理所有细胞(相当于不分批)。用户可结合 pick_batch_size 函数来动态决定批大小。

返回: (row_coph_df, col_coph_df) 二元组, 与标准函数相同格式。需要注意的是, 为了降低内存占用, 函数不会将整个“细胞×细胞”的距离矩阵存于内存, 而是逐簇分批统计, 因此计算得到的结果与正常方法相同, 但使用内存峰值更低。返回的 DataFrame 行列索引为簇类别, 已归一化 [0,1]。

示例:
from sfplot import compute_cophenetic_distances_from_df_memory_opt, pick_batch_size
# 估计合适批大小(假设 df 非常大)
bs = pick_batch_size(n_cells=len(df), dims=2)
row_coph, col_coph = compute_cophenetic_distances_from_df_memory_opt(df, batch_size=bs)

示例依赖: 输入 DataFrame 需满足格式要求。内部实现用到了 sklearn.neighbors.NearestNeighbors 进行最近邻计算, 与标准函数一致。

模块: tbc_analysis_serial.py (串行 Transcript-by-Cell 分析)

该模块提供逐基因逐细胞的空间分析(串行版本), 主要通过一个公开函数 transcript_by_cell_analysis_serial 来执行。它针对空间转录组数据计算每个转录基因相对于细胞簇分布的差异。

函数: _prepare_obs_df(adata, group_df: Optional[pd.DataFrame] = None) -> pd.DataFrame

作用: 内部函数, 从 AnnData 中提取所需的基础 DataFrame(包含 x,y 坐标和 celltype 列)。如果提供了可选的 group_df(含有 cell_id 对应的新分组信息), 则使用其中的 group 列代替默认的 adata.obs["Cluster"] 作为细胞类型; 否则直接使用 Cluster 列。该函数确保返回的 DataFrame 包含每个细胞的坐标和用于距离计算的类别标签。

参数:
- adata: AnnData 对象, 要求 .obsm["spatial"] 存在。函数内部会把 adata.obs["x"] 和 adata.obs["y"] 设置为空间坐标的前两列。
- group_df: 可选的 DataFrame, 含列 cell_id 和 group。若提供, 将按照 cell_id 合并到 adata.obs 以添加/覆盖一个 'group' 列, 用作细胞类型。

返回: 一个 pandas DataFrame, 包含列 ["x","y","celltype"](其中 celltype 列来自 Cluster 或提供的 group)。这个 DataFrame 即后续分析的基础距离计算输入。

示例: (一般不直接调用, 而是在 transcript_by_cell_analysis_serial 内部使用。)

函数: _compute_structure_map(obs_df: pd.DataFrame, out_dir: str, sample: str, method: str = "average") -> pd.DataFrame

作用: 内部函数, 计算给定观测表(细胞坐标与类型表)的 StructureMap(全局行cophenetic距离矩阵), 并将结果保存成热图 PDF 和数值 CSV。换言之, 它计算所有细胞类型簇之间的距离矩阵, 并输出 "StructureMap_of_{sample}.pdf" 以及对应的数值表 "StructureMap_table_{sample}.csv"。返回值则是归一化的行距离矩阵 DataFrame, 供后续分析使用。

参数:
- obs_df: 由 _prepare_obs_df 得到的 DataFrame, 包含所有细胞坐标和类型。
- out_dir: 输出文件目录, 生成的 PDF 和 CSV 会保存在此处。
- sample: 样本名称字符串, 用于文件命名。
- method: cophenetic 距离聚类时使用的 linkage 方法, 默认 "average"。

返回: row_coph: 计算得到的簇间cophenetic距离矩阵(DataFrame形式, 索引为簇名), 已归一化 [0,1]。同时函数有副作用, 会在 out_dir 输出 PDF 图 "StructureMap_of_{sample}.pdf" 以及 CSV "StructureMap_table_{sample}.csv"。

示例: (通常不直接使用, 由 transcript_by_cell_analysis_serial 调用。)

函数: transcript_by_cell_analysis_serial(folder: str, sample_name: Optional[str] = None, output_folder: Optional[str] = None, coph_method: str = "average", df: Optional[pd.DataFrame] = None) -> None

作用: 执行串行的 Transcript-by-Cell 空间分析。给定一个 Xenium 数据文件夹, 该函数会:
1. 加载数据(load_xenium_data) 取得 AnnData 和转录本坐标。
2. 过滤掉无关的转录本(如阴性对照或未指派的 feature)。
3. 准备观测数据表(_prepare_obs_df) 并计算全局 StructureMap(_compute_structure_map) 作为基准。
4. 遍历每个基因, 逐个将该基因的转录本坐标与全局细胞坐标合并, 计算新的 cophenetic 行距离矩阵的该基因行(表示该基因相对于各细胞簇的距离分布), 并将结果逐行写入CSV。
5. 所有基因处理完毕后, CSV 文件包含每个基因相对于各细胞簇的距离度量结果。

由于是串行实现, 该函数适用于调试或小规模数据, 避免了并行的复杂性(如 fork 进程 deadlock)。对于大数据, 可能需要使用并行版以节省时间。

参数:
- folder: Xenium 样本目录路径。
- sample_name: 可选, 自定义样本名, 用于输出文件命名; 如果不提供, 将使用 folder 名称。
- output_folder: 可选, 指定输出结果保存的目录; 默认在当前目录下生成名为 "t_by_c_{sample}" 的文件夹。
- coph_method: cophenetic 距离层次聚类的方法, 默认 "average"。
- df: 可选的 DataFrame(含 cell_id 和 group 列), 用于覆盖默认的聚类。提供该参数时, 将使用 df["group"] 作为细胞类型分类进行分析, 而不使用原始的 Cluster。常用于传入自定义分组(例如手工定义的细胞组)。

返回: 无显式返回。函数运行结束后, 会在指定 output_folder 下生成:
- StructureMap_of_{sample}.pdf: 全局细胞簇结构热图。
- StructureMap_table_{sample}.csv: 对应的簇间距离数值表。
- t_and_c_result_{sample}.csv: 每个基因的分析结果, 行为基因, 列为细胞簇, 值为去除了基因自身后的距离(若基因从未出现则为 NaN)。
控制台也会打印进度和完成信息。

示例:
from sfplot import transcript_by_cell_analysis_serial
# 分析指定 Xenium 数据集(串行处理所有基因)
transcript_by_cell_analysis_serial("data/MyXeniumSample/", sample_name="MySample1")
# 运行结束后, 结果保存在 ./t_by_c_MySample1/ 目录下

示例依赖: 函数内部会调用 load_xenium_data 加载数据, 因此无需手动调用。若提供 df 参数, 则应确保它包含目标 cell_id 列以及对应的 group 列。

模块: tbc_analysis.py (并行 Transcript-by-Cell 分析)

此模块提供多进程并行版的 Transcript-by-Cell 分析 (transcript_by_cell_analysis), 以提高大规模数据运行效率。它通过 Python 的 multiprocessing.Pool 实现, 并利用共享内存 (multiprocessing.shared_memory) 来避免在子进程中重复占用内存。

函数: _init_worker(shm_name, shm_shape, shm_dtype, adata_obs_df, row_coph_df, coph_method: str)

作用: 内部函数, pool worker 初始化函数。在创建进程池时调用, 每个子进程执行此函数以附加到父进程创建的共享内存, 重建全局的转录本坐标 DataFrame, 并复制必要的全局变量。包括:
- 将共享内存块映射为 numpy 数组, 然后构建转录本坐标的 DataFrame _coords_global(含列 x, y, feature_name)。
- 将父进程传来的 adata_obs_df (含细胞 x, y, celltype) 和 row_coph_df(全局 StructureMap) 分别赋给 _adata_obs 和 _row_coph_global。
- 保存 coph_method 供后续计算使用。

这使得每个 worker 子进程拥有只读的转录本全局表和预计算的全局距离矩阵行, 从而在处理各基因时无需重复加载数据或占用额外内存。

参数: (按顺序传入, 不需用户直接调用)
- shm_name, shm_shape, shm_dtype: 共享内存的名称、形状、数据类型, 用于重建 _coords_global DataFrame。
- adata_obs_df: 父进程传来的 DataFrame(细胞坐标+类型)。
- row_coph_df: 父进程传来的全局行距离矩阵 (StructureMap)。
- coph_method: 距离聚类方法字符串。

示例: (内部使用, 不直接调用。)

函数: _process_gene(gene: str) -> Optional[pd.DataFrame]

作用: 内部函数, 在子进程中执行的核心工作函数, 处理单个基因。对于给定基因:
- 如果 _coords_global(全局转录本 DataFrame)中没有该基因的任何转录本坐标:
  - 若该基因在全局 _row_coph_global 距离矩阵中也不存在(从未被标注过), 则返回该基因相对于现有簇的一个空距离行(全 NaN)。
  - 若该基因在 _row_coph_global 中存在(意味着它可能作为细胞簇名出现过), 则直接提取 _row_coph_global 中对应行(并移除自身列)作为结果。
- 如果存在该基因的转录本坐标:
  - 将这些坐标与全局 _adata_obs(细胞坐标+类型)合并, 并将所有该基因的转录点视作一个新的“细胞类型”加入其中。
  - 对合并后的 DataFrame 计算 cophenetic 距离矩阵(使用 _coph_method), 然后取该基因所在行(并移除自身列)作为结果。

函数会捕获所有异常, 保证单个基因计算失败不会中断整个进程, 而是记录错误日志并返回 None(由主进程跳过)。

返回: DataFrame(索引为基因名, 仅一行, 列为细胞簇)或 None(发生错误时)。正常情况下返回的一行表示该基因相对于每个细胞簇的距离度量。

示例: (无需直接调用, 由并行池自动调度执行。)

函数: transcript_by_cell_analysis(folder: str, sample_name: Optional[str] = None, output_folder: Optional[str] = None, coph_method: str = "average", n_jobs: int = 32, maxtasks: int = 50, df: Optional[pd.DataFrame] = None) -> None

作用: 以多进程方式运行 Transcript-by-Cell 空间分析, 与串行版逻辑相同但速度更快。主要流程:
1. 准备输出目录和样本名(与串行版相同)。
2. 调用 load_xenium_data 加载 AnnData, 对应 sample 的 cells 表, 并获取所有转录本坐标(使用 spatialdata_io.xenium, 读取 transcripts 点位)。
3. 过滤掉无关的转录本(如 NegControl 和 Unassigned)。
4. 利用 Python 共享内存将过滤后的转录本坐标数组广播给子进程, 避免重复数据开销。
5. 构建细胞基础表 _adata_obs(同 _prepare_obs_df 功能, 将坐标赋给 obs 并用自定义 df 或默认 Cluster 列作为 celltype)。
6. 计算全局 StructureMap 距离矩阵 _row_coph_global 并保存 PDF/CSV(类似 _compute_structure_map)。
7. 使用 multiprocessing.Pool 创建进程池, 初始化 _init_worker, 并通过 pool.imap_unordered 将所有基因列表分配给 _process_gene 并行处理。
8. 将每个 _process_gene 返回的结果 DataFrame 逐条写入输出 CSV "t_and_c_result_{sample}.csv"(与串行版结果格式相同)。处理过程中用 tqdm 输出进度条。
9. 全部完成后, 清理进程池和共享内存, 提示完成。

参数:
- folder, sample_name, output_folder, coph_method, df: 含义与串行版 transcript_by_cell_analysis_serial 相同。
- n_jobs: 并行进程数, 默认 32。建议根据CPU核心数和内存大小设置。
- maxtasks: 每个子进程处理的最大任务数, 默认 50。此参数可控制在处理一定数量任务后重启子进程, 以释放内存(避免单个进程处理过多基因导致内存占用增长)。

返回: 无直接返回。执行完毕后, 在 output_folder 下产生的文件与串行版相同(StructureMap PDF/CSV和最终结果CSV)。不同的是并行版会在开始时输出一条 "Workers initialized, start processing genes ..." 信息, 随后显示一个进度条以指示基因处理进度。

示例:
from sfplot import transcript_by_cell_analysis
# 并行分析 Xenium 数据集, 提高处理大样本(上万个基因)的速度
transcript_by_cell_analysis("data/MyXeniumSample/", sample_name="MySample1", n_jobs=16, maxtasks=100)
# 结果将保存在 ./t_by_c_MySample1/ 目录, 与串行版本格式相同

示例依赖: 需要操作系统支持 multiprocessing 共享内存(Python 3.8+)。函数内部大量使用全局变量在子进程中的共享, 建议仅通过 CLI 或脚本调用, 不要在交互式环境多次重复调用同一进程中, 否则需注意释放资源。

模块: transcript_analysis.py

函数: process_gene(gene, adata, transcripts_pdf) -> pd.DataFrame

作用: 给定单个基因名称, 提取该基因的所有转录本坐标并与现有细胞数据合并, 计算该基因相对于每个细胞类型簇的cophenetic距离行。该函数实现与前述 transcript_by_cell_analysis_serial 类似的单基因处理:
- 从 adata.obs 获取细胞坐标和类型 (使用列 x, y, celltype)。
- 从 transcripts_pdf DataFrame 中筛选出指定基因的转录本坐标(假定 transcripts_pdf 含列 x, y, feature_name, 其中 feature_name == gene)。并将该 DataFrame 的 feature_name 列重命名为 celltype。
- 将上述细胞 DataFrame 与转录本 DataFrame 纵向合并, 然后调用 compute_cophenetic_distances_from_df 计算距离矩阵(此时合并后的数据包含原细胞簇+一个基因簇)。
- 提取结果的行距离矩阵中该基因所在行(并移除自身列), 形成一个仅包含该基因对各细胞簇距离的 DataFrame(一行)。返回这个 DataFrame。

参数:
- gene: 目标基因名称(与 transcripts_pdf 中 feature_name 对应)。
- adata: AnnData 对象, 用于提供细胞坐标和类型(adata.obs 应含 x, y, celltype 列)。
- transcripts_pdf: DataFrame, 包含转录本点坐标信息, 至少有 x, y, feature_name 列, 其中 feature_name 标识基因。通常通过 Xenium 的 sdata.points["transcripts"] 获得并转成 pandas。

返回: 包含一行的数据框, 索引为该基因, 列为细胞类型, 各值为该基因与对应细胞类型的距离(0表示完全重合, 1为最远, 已归一化)。注意若计算过程中该基因作为新的“簇”加入没有与自身比较, 因此结果表中不应含该基因自身列。

示例:
# 假设 adata 和 transcripts_pdf 已经定义
df_gene = process_gene("MYC", adata, transcripts_pdf)
print(df_gene.loc["MYC"])  # 输出 MYC 基因相对于各细胞簇的距离值

示例依赖: 需先获取 adata (例如通过 load_xenium_data) 以及对应的 transcripts_pdf(例如通过 spatialdata_io.xenium(...).points["transcripts"] 转为 pandas DataFrame)。本函数适合小规模并行(可手动用 joblib Parallel 调度多个基因), 而大规模分析建议使用模块 tbc_analysis.py 中的批量函数。

模块: plotting.py

plotting.py 模块提供了一些快速可视化函数, 将距离计算与绘图结合, 方便对单个样本进行一键式分析。它们内部实际上重复了部分 cophenetic 距离计算逻辑, 但对于只关注最终图形的情况, 可以直接使用这些函数。

函数: generate_cluster_distance_heatmap_from_path(base_path: str, sample: str, figsize: tuple = (8, 8), output_dir: Optional[str] = None, show_dendrogram: bool = True) -> None

作用: 直接从数据路径生成簇距离热图 PDF。给定数据所在的基本路径和样本名称, 该函数会:
- 构建样本文件夹路径并调用 load_xenium_data 读取数据(使用默认 Cluster 聚类)。
- 提取所有细胞的空间坐标 (adata.obsm["spatial"]) 和聚类标签 (adata.obs["Cluster"]), 计算每个细胞到各簇中心的最近距离(形成 cell × cluster 矩阵)。
- 对上述距离矩阵按细胞簇取均值, 得到 cluster × cluster 平均距离矩阵, 并使用 seaborn.clustermap 进行聚类绘制。
- 将热图保存为 PDF 文件, 默认命名为 "SFplot_of_{sample}.pdf"(SFplot 即结构距离图)。

这个函数等价于将数据加载、计算距离、绘制热图合为一步, 适合快速检查结果。(注意: 没有返回值, 结果直接保存文件。)

参数:
- base_path: 基础目录路径, 实际数据文件夹路径将是 base_path/sample。
- sample: 样本文件夹名称。
- figsize: 输出图大小(默认 8x8 英寸)。
- output_dir: 输出目录, 默认 None 表示当前工作目录。
- show_dendrogram: 是否绘制行列树状图, 默认 True。如设 False, 将跳过调整树状图布局的步骤。

返回: 无返回。调用后会在 output_dir 下生成文件 SFplot_of_{sample}.pdf。

示例:
from sfplot import generate_cluster_distance_heatmap_from_path
generate_cluster_distance_heatmap_from_path("/data/projectX/", "Sample1", output_dir="./figs")
# 运行结束后, 会在 ./figs/SFplot_of_Sample1.pdf 得到簇距离热图

示例依赖: 内部调用 load_xenium_data, 需要目录结构正确且安装了相关依赖。无需用户手动计算距离或调用其他函数。

函数: generate_cluster_distance_heatmap_from_adata(adata: anndata.AnnData, cluster_col: str = "Cluster", output_dir: Optional[str] = None, output_filename: Optional[str] = None, figsize: tuple = (8, 8), cmap: str = "RdBu", max_scale: float = 10, show_dendrogram: bool = True) -> None

作用: 与上述函数类似, 但直接从 AnnData 对象生成热图。适合已经通过其他方式获取了 AnnData(并可能对数据做了预处理)的情况。函数执行步骤与 from_path 类似:
- 使用给定的 cluster_col 作为类别, 将所有细胞到各簇最近距离计算出来。
- 计算 cluster × cluster 平均距离矩阵, 绘制聚类热图并保存。

相较于 from_path, 这个函数允许用户自定义 colormap, 控制 Anndata 数据预处理(例如 adata 可以自行决定是否标准化)等细节。

参数:
- adata: AnnData 对象, 需要包含 .obsm["spatial"] (坐标)和 adata.obs[cluster_col] (类别标签), 以及 adata.obs["cell_id"] (用于索引)。
- cluster_col: 使用的簇标签列名, 默认 "Cluster"。
- output_dir: 输出目录, 默认当前目录。
- output_filename: 输出文件的名称。如果未指定, 将使用 "clustermap_output_{sample}.pdf" 格式 (sample 名读取自 adata.uns["sample"], 如果有的话, 否则用 "Sample")。
- figsize: 热图的大小, 默认 (7, 7)。
- cmap: 热图的颜色映射, 默认 "RdBu"。
- max_scale: sc.pp.scale 的 max_value 参数, 用于裁剪 Z-score, 默认为 10。(该参数在本函数实现中未显式用到, 一般忽略)
- show_dendrogram: 是否绘制 dendrogram, 默认 True。

返回: 无返回。结果图保存为指定文件。成功执行后, 控制台会打印 "Cluster distance heatmap saved to ..."

示例:
from sfplot import generate_cluster_distance_heatmap_from_adata
# 假设 adata 已包含需要的 cluster 列和 spatial 坐标
generate_cluster_distance_heatmap_from_adata(adata, cluster_col="Cluster", output_dir="./figs", output_filename="myplot.pdf")

示例依赖: adata 对象准备完备。若 adata.uns["sample"] 没有样本名且未提供 output_filename, 默认文件名会使用 "SFplot_of_Sample.pdf"。

函数: generate_cluster_distance_heatmap_from_df(df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y', celltype_col: str = 'celltype', sample: str = 'Sample', output_dir: Optional[str] = None, output_filename: Optional[str] = None, figsize: tuple = (8, 8), cmap: str = "RdBu", show_dendrogram: bool = True) -> None

作用: 从原始 DataFrame 直接计算簇平均距离并绘制热图。接口与上面类似, 只是数据来源换成了一个 DataFrame: 需要提供坐标和细胞类型列。函数内部会:
- 计算每个细胞到各簇最近距离矩阵(cell × cluster)。
- 按簇求平均得到 cluster × cluster 距离矩阵, 删除全NaN列后进行聚类热图绘制。
- 保存 PDF(默认名 "clustermap_output.pdf" 或指定名称)。

此函数与 compute_cophenetic_distances_from_df+绘图 的区别在于: 它直接使用简单的最近邻+均值来构造距离, 而 cophenetic 函数是经过层次聚类再计算 cophenetic 距离, 理论上结果会略有不同。但对于大体观察分布, 此函数更加简明快速。

参数:
- df: 输入 DataFrame, 要求包含指定的坐标列和类别列。
- x_col, y_col: 坐标列名, 默认为 'x' 和 'y'。
- celltype_col: 类别列名, 默认为 'celltype'。
- sample: 样本名称, 用于图标题中(默认 "Sample")。
- output_dir, output_filename, figsize, cmap, show_dendrogram: 含义和默认值同上, 不再赘述。output_filename 未提供时, 此函数默认为 "clustermap_output.pdf"。

返回: 无返回。聚类热图 PDF 文件保存在指定路径下。

示例:
from sfplot import generate_cluster_distance_heatmap_from_df
# DataFrame df 包含 x,y,celltype 列
generate_cluster_distance_heatmap_from_df(df, sample="TestSample", output_dir="./", output_filename="TestSample_heatmap.pdf")

示例依赖: DataFrame 准备完备。此函数不依赖 AnnData, 可用于任意有坐标和类别的数据情景。

模块: sfplot/gui/gui_app.py (GUI 图形界面)

GUI 部分提供了一个基于 Tkinter 的桌面应用程序界面, 用于通过交互方式加载数据和生成 SFplot 热图。主要包括启动函数和 MainApp 主窗口类。

函数: resource_path(relative_path: str) -> str

作用: 用于获取资源文件的绝对路径的辅助函数。在应用被 PyInstaller 冻结为可执行文件时, 资源文件(如图像)会被打包到临时目录, 此函数可统一路径获取逻辑。它尝试从 sys._MEIPASS(PyInstaller专用临时目录)寻找资源, 否则退回当前脚本所在目录。

参数: relative_path: 资源文件的相对路径。
返回: 资源文件的绝对路径字符串。

示例: (通常在内部使用)
logo_path = resource_path("assets/logo.png")

函数: main() -> None

作用: GUI 应用程序入口。调用该函数会创建并运行主窗口 MainApp。相当于 Tkinter 应用的 mainloop 启动函数。

示例:
from sfplot.gui.gui_app import main
main()  # 启动 GUI 程序

调用后会出现一个窗口界面, 包括后述的 CSV Heatmap 和 Xenium Heatmap 两个功能选项卡。

类: MainApp (tk.Tk 派生)

作用: 定义了 SFplot 图形界面的主窗口及其交互逻辑。主界面包含两个标签页(tab): CSV Heatmap 和 Xenium Heatmap, 分别对应加载普通 CSV 数据和加载 Xenium 数据进行分析。MainApp 初始化时会构建整个界面的控件, 并设置一系列内部方法作为回调函数处理用户交互。

主要属性和界面元素:
- 两个 Tab 页: tab_csv 和 tab_xenium, 分别用于CSV文件分析和Xenium数据分析。
- 各 Tab 内的控件: 比如 CSV Tab 包含“Select CSV File”按钮(load_btn)、“Plot CSV Heatmap” 按钮(draw_btn)、缩放下拉菜单、进度条、日志文本框(log_text) 和显示区(display_frame)等。Xenium Tab 则有“Select Xenium Dir”按钮、“Load Xenium Data”按钮(load_xenium_btn)、“Select Selection CSV”按钮(selcsv_btn)、“Plot Xenium Heatmap”按钮(plot_x_btn), 以及对应的缩放控件、进度条、日志框(log_text2) 和显示区(display_frame2)。
- 队列 _queue: queue.Queue 对象, 用于在后台线程与主线程间传递消息。消息类型包括 "log" 日志信息、"progress" 进度百分比、"image" 生成的PIL图像、"error" 错误信息, Xenium分支也有 "x_log", "x_image" 等。

主要方法(按交互逻辑分类):
- CSV 模式:
  - _ask_csv_file(): 弹出文件选择对话框, 让用户选择一个包含坐标和类别信息的 CSV 文件。成功选择后, 记录路径到 csv_path, 日志提示, 并启用 “Plot CSV Heatmap” 按钮。
  - _start_csv_worker(): 点击 “Plot CSV Heatmap” 时调用。禁用相关按钮, 启动一个后台线程执行 _csv_worker()。
  - _csv_worker(): 后台线程实际执行的函数。流程: 读取 CSV -> 计算 cophenetic 距离矩阵(compute_cophenetic_distances_from_df) -> 生成热图 PIL 图像(plot_cophenetic_heatmap(return_image=True)) -> 通过 _queue 发送 "image" 消息传回主线程显示。执行过程中按阶段发送 "progress" 和 "log" 更新进度条和日志。若发生异常, 则发送 "error" 消息。
  - _on_scale_change_csv(): 当用户调整缩放比例下拉菜单时调用, 如果有已显示的图像 _orig_img, 则重新调用 _display_csv_image 按新的比例缩放显示。

- Xenium 模式:
  - _ask_xenium_dir(): 弹出目录选择对话框, 让用户选 Xenium 数据文件夹。选定后保存路径到 xenium_path, 日志提示, 并启用 “Load Xenium Data” 按钮。
  - _load_xenium_data(): 点击 “Load Xenium Data” 时调用。禁用按钮并启动后台线程 _xenium_load_worker()。
  - _xenium_load_worker(): 后台线程, 调用 load_xenium_data(xenium_path, normalize=False) 加载 Xenium 数据(不归一化), 将结果 AnnData 缓存在 adata_cache。完成后通过 _queue 发送 "x_log" 日志和 "x_enable_csv" 消息, 通知主线程启用 “Select Selection CSV” 按钮。
  - _ask_selection_csv(): 弹出文件对话框选择一个包含 选择集(selection) 的CSV(例如用户选定的一批目标 cell_id 列表)。选定后保存路径到 selection_csv, 日志提示, 并启用 “Plot Xenium Heatmap” 按钮。
  - _start_xenium_plot(): 点击 “Plot Xenium Heatmap” 时调用, 禁用相关按钮并启动后台线程 _xenium_plot_worker()。
  - _xenium_plot_worker(): 后台线程, 将 selection_csv 读入 DataFrame, 提取其中的细胞ID列表, 在 adata_cache 中筛选对应子集(AnnData 包含这些 cell_id), 然后计算 cophenetic 距离矩阵(compute_cophenetic_distances_from_adata) -> 生成热图 PIL 图像(plot_cophenetic_heatmap(return_image=True)) -> 通过 _queue 发送 "x_image" 消息以显示热图。同时发送 "x_enable_csv" 重新启用 Xenium Tab 中的按钮。异常则发送 "x_error"。
  - _on_scale_change_x(): Xenium Tab 下调整缩放时调用, 如果有已显示图像 _orig_img2 则调用 _display_x_image 更新显示。

- 通用:
  - _poll_queue(): 主线程定时调用(通过 self.after(100, ...) 计划每100ms运行一次), 检查 _queue 中的消息并分类处理:
    - "log" 和 "x_log": 调用 _log_csv 或 _log_x 将文本追加到日志区域。
    - "progress": 更新 CSV 进度条和百分比标签。
    - "image": 调用 _display_csv_image 显示新的热图。
    - "error": 弹出错误消息对话框, 并(可以考虑)重新启用按钮。
    - "done": 在 CSV 分析完成后, 重启用 “Select CSV File” 与 “Plot CSV Heatmap” 按钮; 在 Xenium 加载完成后(通过同一 tag 名, 可能是代码的小冗余)启用 Xenium 加载按钮。处理完所有消息后, 函数通过 after 自调, 实现循环监听。
    - "x_progress": 更新 Xenium 进度条(当前实现中 _xenium_load_worker 未发送此类型, 保留以扩展)。
    - "x_image": 调用 _display_x_image 显示 Xenium 热图。
    - "x_error": 弹出错误消息。
    - "x_enable_csv": 重新启用 Xenium Tab 中 “Select Selection CSV” 和 “Plot Xenium Heatmap” 按钮(用于在 Xenium热图绘制完成后允许用户加载新的 selection CSV)。
  - _log_csv(msg: str) 和 _log_x(msg: str): 将消息追加到对应日志文本框中(CSV或Xenium), 并确保滚动到底部。
  - _display_csv_image(pil_img: Image.Image) 和 _display_x_image(pil_img: Image.Image): 在界面右侧的显示区域创建 Canvas, 将给定的 PIL 图像缩放适配框大小并显示。函数实现了图像的缩放(根据当前 scale 变量)、滚动条支持, 以及在每次显示新图前销毁先前的图像 frame 以释放资源。

- 使用顺序:
  1. 启动程序后, 选择相应Tab。
  2. CSV 模式下: 先点击 “Select CSV File” -> 选择含数据的CSV(需至少有 x,y,celltype 列) -> 点击 “Plot CSV Heatmap” 开始分析绘图。完成后热图会显示在界面右侧。(注: CSV 分析流程完全在内存中进行。)
  3. Xenium 模式下: 点击 “Select Xenium Dir” -> 选择 Xenium 数据文件夹 -> 点击 “Load Xenium Data” 加载数据(需一定时间) -> 选择一个 Selection CSV(包含要关注的一组细胞ID, 比如某类细胞清单) -> 点击 “Plot Xenium Heatmap” 计算该选择集的结构距离图。热图生成后会显示。(注: Xenium 模式下, 全局的 StructureMap 会在后台加载数据后自动计算一次, 生成的数据缓存在内存, 以便不同 selection 重复利用。)

示例: GUI 应用通常通过命令行调用包的入口点或直接运行脚本启动。例如, 在终端运行:
$ sfplot
或在 Python 中执行 sfplot.cli.main()(如果安装了 Typer CLI)。这将调用 gui_app.main() 打开 GUI。在GUI中按照提示按钮逐步操作即可, 无需显式调用上述内部方法。
