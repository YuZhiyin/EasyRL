import pandas as pd
import json

# 1. 读取 parquet 文件
df = pd.read_parquet("deepmath_all_new.parquet")

# 2. 提取字段
data = []
for i, row in df.iterrows():
    try:
        # 从 prompt 中提取 question
        prompt_data = row["prompt"]
        if isinstance(prompt_data, str):
            prompt_data = json.loads(prompt_data)
        question = prompt_data[0]["content"] if prompt_data else ""

        # 从 reward_model 中提取 answer
        reward_data = row["reward_model"]
        if isinstance(reward_data, str):
            reward_data = json.loads(reward_data)
        answer = reward_data.get("ground_truth", "")

        # 难度
        difficulty = row["extra_info"].get("difficulty", None)
        if isinstance(difficulty, str):
            try:
                difficulty = json.loads(difficulty).get("difficulty", None)
            except:
                pass

        data.append({
            "id": len(data),
            "question": question,
            "answer": answer,
            "difficulty": difficulty
        })
    except Exception as e:
        print(f"Error at row {i}: {e}")

# 3. 保存为 CSV
out_df = pd.DataFrame(data)
out_df.to_csv("deepmath_all_new.csv", index=False, encoding="utf-8-sig")

print(f"✅ 转换完成，共 {len(out_df)} 条数据，已保存到 deepmath_all_new.csv")
