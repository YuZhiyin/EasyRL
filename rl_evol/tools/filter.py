import pandas as pd

df1 = pd.read_csv("/mnt/shared-storage-user/yuzhiyin/RLEvol_new/save_deepmath/llama/llama3bround1left.csv")
df2 = pd.read_csv("/mnt/shared-storage-user/yuzhiyin/RLEvol_new/save_deepmath/llama/deepmath_pseudo_1105_1847-round2new.csv", encoding='utf-8-sig')

# 统一列名和类型
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df1['id'] = df1['id'].astype(str).str.strip()
df2['id'] = df2['id'].astype(str).str.strip()

# 检查公共id
common_ids = set(df1['id']) & set(df2['id'])
print("公共ID数:", len(common_ids))

# 过滤出不在df2中的行
df3 = df1[~df1['id'].isin(df2['id'])]

# extra_ids = set(df2['id']) - set(df1['id'])
# print(extra_ids)
df3.to_csv("llama3bround2left.csv", index=False)

print("df1行数:", len(df1))
print("df2行数:", len(df2))
print("df3行数:", len(df3))



