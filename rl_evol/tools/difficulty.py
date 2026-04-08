import pandas as pd
import ast

# 读取两个文件
df_csv = pd.read_csv("deepmath_pseudo_1109_2116-round3all.csv")
df_parquet = pd.read_parquet("deepmath_unlabel.parquet")

# 从 extra_info 提取 index 和 difficulty
df_parquet["extra_info"] = df_parquet["extra_info"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_parquet["index"] = df_parquet["extra_info"].apply(lambda d: d.get("index"))
df_parquet["difficulty"] = df_parquet["extra_info"].apply(lambda d: d.get("difficulty"))

# 只保留映射需要的列
df_map = df_parquet[["index", "difficulty"]]

# 合并：id 对应 index
df_merged = df_csv.merge(df_map, left_on="id", right_on="index", how="left")

# 去掉冗余列 index（可选）
df_merged = df_merged.drop(columns=["index"])

# # 保存
# df_merged.to_csv("deepmath_pseudo_0918_2227_with_difficulty.csv", index=False)

mean_difficulty = df_merged["difficulty"].mean()
print("difficulty 平均值:", mean_difficulty)

print("合并完成，新增 difficulty 列。")
