import os
import shutil
import tempfile
from pathlib import Path
from spatialdata_io import visium  # 假设 spatialdata_io.visium 函数已正确导入


def read_visium_bin(base_path: str, dataset_id: str = None,  # 保持原接口参数
                    counts_file: str = 'filtered_feature_bc_matrix.h5',
                    tissue_positions_file: str = None,
                    fullres_image_file: str = None,
                    scalefactors_file: str = None,
                    var_names_make_unique: bool = True,
                    imread_kwargs=None, image_models_kwargs=None,
                    keep_tmp: bool = False):
    """
    读取10x Genomics Visium数据（使用HDF5矩阵文件），返回SpatialData对象。
    修改：使用临时影子目录解决tissue_positions文件路径限制问题。
    """
    # 1. 创建临时影子目录及spatial子目录
    shadow_dir = Path(tempfile.mkdtemp())  # 创建临时目录
    spatial_shadow = shadow_dir / "spatial"
    spatial_shadow.mkdir(parents=True, exist_ok=True)
    try:
        # 2. 准备tissue_positions.csv文件内容
        tissue_file_path = None
        if tissue_positions_file:
            tissue_file_path = Path(tissue_positions_file)
        else:
            # 尝试在base目录下找到tissue_positions文件
            for fname in ["tissue_positions.csv", "tissue_positions_list.csv"]:
                candidate = Path(base_path) / "spatial" / fname
                if candidate.exists():
                    tissue_file_path = candidate
                    break
        if tissue_file_path is None or not tissue_file_path.exists():
            shutil.rmtree(shadow_dir)  # 清理临时目录
            raise FileNotFoundError(f"无法找到组织位置信息文件（tissue_positions）于 {base_path} 或指定路径。")
        # 将tissue_positions内容写入影子目录中的 spatial/tissue_positions.csv
        shadow_tissue_csv = spatial_shadow / "tissue_positions.csv"
        shutil.copyfile(tissue_file_path, shadow_tissue_csv)

        # 3. 链接或复制必要的文件到影子目录
        base_path = Path(base_path)
        # 3a. 处理表达矩阵（counts）文件
        counts_path = base_path / (dataset_id + "_" + counts_file if dataset_id else counts_file)
        if not counts_path.exists():
            # counts文件找不到
            shutil.rmtree(shadow_dir)
            raise FileNotFoundError(f"无法在 {base_path} 找到计数矩阵文件 {counts_path.name}")
        # 将counts文件链接/复制到影子目录根下
        shadow_counts_path = shadow_dir / counts_path.name
        try:
            # 首选创建符号链接
            os.symlink(counts_path, shadow_counts_path)
        except Exception:
            # 如不支持符号链接则尝试复制
            shutil.copy2(counts_path, shadow_counts_path)

        # 3b. 处理spatial子目录下的其他文件（图像和scalefactors等）
        spatial_files = []
        # 如果用户指定了fullres_image_file或scalefactors_file路径，则也添加
        if fullres_image_file:
            spatial_files.append(Path(fullres_image_file))
        if scalefactors_file:
            spatial_files.append(Path(scalefactors_file))
        # 添加常见的文件名，如果存在则复制/链接
        default_spatial_names = ["tissue_hires_image.png", "tissue_lowres_image.png", "scalefactors_json.json"]
        for name in default_spatial_names:
            file_path = base_path / "spatial" / name
            if file_path.exists():
                spatial_files.append(file_path)
        # 去重并复制/链接文件
        spatial_files = set(spatial_files)
        for file_path in spatial_files:
            target_path = spatial_shadow / file_path.name
            try:
                os.symlink(file_path, target_path)
            except Exception:
                shutil.copy2(file_path, target_path)

        # 4. 调用visium读取数据，指定tissue_positions_file相对路径
        sdata = visium(
            path=shadow_dir,
            dataset_id=dataset_id,
            counts_file=shadow_counts_path.name,
            fullres_image_file=fullres_image_file if fullres_image_file else None,
            tissue_positions_file="spatial/tissue_positions.csv",  # 相对路径
            scalefactors_file=Path("spatial") / "scalefactors_json.json" if (
                    spatial_shadow / "scalefactors_json.json").exists() else None,
            var_names_make_unique=var_names_make_unique,
            imread_kwargs=imread_kwargs or {},
            image_models_kwargs=image_models_kwargs or {}
        )
        return sdata  # 返回SpatialData对象
    finally:
        # 5. 清理临时目录（除非用户要求保留）
        if not keep_tmp:
            shutil.rmtree(shadow_dir, ignore_errors=True)
