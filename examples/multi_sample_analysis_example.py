"""
Example: average cophenetic distances across multiple spatial samples.

Usage
-----
Replace ``adata1``, ``adata2``, ``adata3`` with your own AnnData objects,
update ``output_dir``, then run the script.
"""

from sfplot import compute_cophenetic_distances_from_adata

# Replace with your actual AnnData objects
adata_list = [adata1, adata2, adata3]  # noqa: F821

# Initialize lists for storing results
row_cophenetic_list = []
col_cophenetic_list = []

# Iterate over each adata object and compute normalized cophenetic distance matrices
for adata in adata_list:
    row_norm, col_norm = compute_cophenetic_distances_from_adata(adata, cluster_col="Cluster", output_dir="your/output/path")
    row_cophenetic_list.append(row_norm)
    col_cophenetic_list.append(col_norm)

# If each result DataFrame has the same row/column labels, sum and average directly:
n = len(adata_list)
average_row_cophenetic = sum(row_cophenetic_list) / n
average_col_cophenetic = sum(col_cophenetic_list) / n

# Output or save the averaged results
print("Average Row Cophenetic Distance:")
print(average_row_cophenetic)

print("Average Column Cophenetic Distance:")
print(average_col_cophenetic)
