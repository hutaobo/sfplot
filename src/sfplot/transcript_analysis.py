import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def process_gene(gene, adata, transcripts_pdf):
    df_obs = adata.obs[['x', 'y', 'celltype']]
    df_transcripts = transcripts_pdf[transcripts_pdf['feature_name'] == gene][['x', 'y', 'feature_name']]
    df_transcripts = df_transcripts.rename(columns={'feature_name': 'celltype'})

    df = pd.concat([df_obs, df_transcripts], ignore_index=True)

    row_coph, col_coph = compute_cophenetic_distances_from_df(
        df,
        x_col='x',
        y_col='y',
        celltype_col='celltype',
        output_dir=None,
        method='average'
    )

    row_series = row_coph.loc[gene].drop(labels=gene, errors='ignore')

    row_df = pd.DataFrame(row_series).T
    row_df.index = [gene]

    return row_df

df_obs = adata.obs[['x', 'y', 'celltype']]
df_transcripts = transcripts_pdf[transcripts_pdf['feature_name'] == 'AAMP'][['x', 'y', 'feature_name']]
df_transcripts = df_transcripts.rename(columns={'feature_name': 'celltype'})

# 并行计算
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(process_gene)(gene, adata, transcripts_pdf)
    for gene in tqdm(adata.var.index, desc="Processing genes")
)

# 合并所有结果为一个DataFrame
result = pd.concat(results, axis=0)
