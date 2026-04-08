import pandas as pd 

# 读取 CSV 文件
df = pd.read_csv("deepmath_pseudo_1106_0932-round3new.csv")

print(f"📊 总共有 {len(df)} 个 id（即行数）")

# 统计 PseudoLabel 为空的行数（包括 NaN 或空字符串）
empty_count = df["PseudoLabel"].isna().sum() + (df["PseudoLabel"].astype(str).str.strip() == "").sum()
print(f"🔍 'PseudoLabel' 列中为空的行数: {empty_count}")

# 去掉 PseudoLabel 为空的行
df_cleaned = df[~(df["PseudoLabel"].isna() | (df["PseudoLabel"].astype(str).str.strip() == ""))]

print(f"✅ 清洗后还剩 {len(df_cleaned)} 行")

# 保存为新的 CSV 文件
df_cleaned.to_csv("llama-round3-new.csv", index=False, encoding="utf-8-sig")
print("💾 已保存清洗后的文件：mathscale_pseudo_cleaned.csv")


# # 只保留 question 和 PseudoLabel 两列
# df_new = df[["question", "PseudoLabel"]]

# # 保存到新的 CSV 文件
# df_new.to_csv("test_filtered.csv", index=False)


