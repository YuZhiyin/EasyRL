import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset


# Step 1: 读取两个 CSV 文件
df1 = pd.read_csv("deepmath_label_filtered.csv")  # 你想追加的
df2 = pd.read_csv("llama-round3-new.csv")  # 原来的


# Step 2: 合并 DataFrame
df_all = pd.concat([df1, df2], ignore_index=True)

# Step 3: 转换为 HuggingFace Dataset
dataset = Dataset.from_pandas(df_all)

# Step 4: 格式化为需要的结构
def transform_data(example, idx):
    suffix = " Let's think step by step and output the final answer within \\boxed{}."
    
    new_prompt = [{
        "role": "user",
        "content": example["question"] + suffix
    }]

    reward_model = {
        "ground_truth": example["PseudoLabel"],
        "style": "rule"
    }

    extra_info = {
        # "difficulty": example["difficulty"],
        "index": idx,
        "name": "numina_math",
        "split": "train"
    }

    return {
        "prompt": new_prompt,
        "reward_model": reward_model,
        "data_source": "numina_math",
        "extra_info": extra_info
    }

formatted_data = dataset.map(transform_data, with_indices=True, remove_columns=dataset.column_names)

print(f"📊 formatted_data 一共有 {len(formatted_data)} 条")

formatted_data.to_parquet("llama3b-round3-new.parquet")
