import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors


def compute_groupwise_average_distance_between_two_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_x_col: str = 'x',
    df1_y_col: str = 'y',
    df1_celltype_col: str = 'celltype',
    df2_x_col: str = 'x',
    df2_y_col: str = 'y',
    df2_celltype_col: str = 'celltype',
    n_jobs: int = -1  # number of jobs for parallel computation; -1 uses all CPUs
) -> pd.DataFrame:
    """
    Compute and return a matrix where rows represent each unique source cell type in df1,
    columns represent each unique target cell type in df2,
    and each element is the average nearest-neighbor distance from all cells in the corresponding
    source group in df1 to all cells in the corresponding target group in df2.

    Parameters:
    ----------
    df1 : pd.DataFrame
        DataFrame containing source cell data.
    df2 : pd.DataFrame
        DataFrame containing target cell data.
    df1_x_col : str, optional
        Column name for x coordinates in df1. Defaults to 'x'.
    df1_y_col : str, optional
        Column name for y coordinates in df1. Defaults to 'y'.
    df1_celltype_col : str, optional
        Column name for cell type in df1. Defaults to 'celltype'.
    df2_x_col : str, optional
        Column name for x coordinates in df2. Defaults to 'x'.
    df2_y_col : str, optional
        Column name for y coordinates in df2. Defaults to 'y'.
    df2_celltype_col : str, optional
        Column name for cell type in df2. Defaults to 'celltype'.
    n_jobs : int, optional
        Number of parallel jobs. Defaults to -1 (use all CPUs).

    Returns:
    -------
    pd.DataFrame
        A matrix with unique source cell types from df1 as row index,
        unique target cell types from df2 as column index, and average nearest-neighbor
        distance for each corresponding group pair as values.
    """
    # Check whether df1 and df2 contain the required columns
    required_df1_columns = {df1_x_col, df1_y_col, df1_celltype_col}
    required_df2_columns = {df2_x_col, df2_y_col, df2_celltype_col}
    if not required_df1_columns.issubset(df1.columns):
        raise ValueError(f"df1 must contain the following columns: {required_df1_columns}")
    if not required_df2_columns.issubset(df2.columns):
        raise ValueError(f"df2 must contain the following columns: {required_df2_columns}")

    # Extract source and target cell types
    source_cell_types = df1[df1_celltype_col].unique()
    target_cell_types = df2[df2_celltype_col].unique()

    # Initialize result matrix
    average_distance_matrix = pd.DataFrame(index=source_cell_types, columns=target_cell_types, dtype=float)

    # Define inner function for parallel computation of a single source cell type row
    def compute_for_source(src_type):
        df1_subset = df1[df1[df1_celltype_col] == src_type]
        source_coords = df1_subset[[df1_x_col, df1_y_col]].values
        row_result = {}

        if source_coords.shape[0] == 0:
            for tgt_type in target_cell_types:
                row_result[tgt_type] = np.nan
            return src_type, row_result

        for tgt_type in target_cell_types:
            df2_subset = df2[df2[df2_celltype_col] == tgt_type]
            target_coords = df2_subset[[df2_x_col, df2_y_col]].values
            if target_coords.shape[0] == 0:
                row_result[tgt_type] = np.nan
                continue
            # Build nearest-neighbor model and compute distances from source_coords to nearest target_coords
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
            nbrs.fit(target_coords)
            distances, _ = nbrs.kneighbors(source_coords)
            average_distance = distances[:, 0].mean()
            row_result[tgt_type] = average_distance

        return src_type, row_result

    # Compute rows for all source cell types in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute_for_source)(src_type) for src_type in source_cell_types)

    # Write computed results into result matrix
    for src_type, row_result in results:
        for tgt_type, avg_dist in row_result.items():
            average_distance_matrix.loc[src_type, tgt_type] = avg_dist

    return average_distance_matrix
