# 假设你有多个 adata 对象，存放在列表中
adata_list = [adata1, adata2, adata3]  # 替换为你的实际对象

# 初始化用于存储结果的列表
row_cophenetic_list = []
col_cophenetic_list = []

# 遍历每个 adata 对象，调用函数计算归一化后的 cophenetic 距离矩阵
for adata in adata_list:
    row_norm, col_norm = compute_cophenetic_distances_from_adata(adata, cluster_col="Cluster", output_dir="your/output/path")
    row_cophenetic_list.append(row_norm)
    col_cophenetic_list.append(col_norm)

# 如果确保每个结果的 DataFrame 具有相同的行、列标签，则可以直接相加求平均：
n = len(adata_list)
average_row_cophenetic = sum(row_cophenetic_list) / n
average_col_cophenetic = sum(col_cophenetic_list) / n

# 输出或保存平均结果
print("Average Row Cophenetic Distance:")
print(average_row_cophenetic)

print("Average Column Cophenetic Distance:")
print(average_col_cophenetic)
