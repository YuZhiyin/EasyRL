import os 
import fire
import pandas as pd
import random
import copy
import numpy as np
import datetime
from tqdm import tqdm
from EasyRL.RL.RLEvol.eval import Evaluator, EvalConfig
from EasyRL.RL.RLEvol.config import TASK_CONFIG, MODELS_CONFIG
from math_verify import parse, verify
from EasyRL.RL.RLEvol.common import format_question_math, extract_answer, clear_mem


def configure_model(model: str) -> dict:
    """Configure model environment and API keys"""
    if model in MODELS_CONFIG:
        config = MODELS_CONFIG[model]
        if 'url' in config:
            os.environ['LLM_BASE_URL'] = config['url']
        if 'OPENAI_API_KEY' in config:
            os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
        return config
    return {'name': model}


def load_data(csv_path: str) -> list:
    """Load DeepMath CSV and return sample list"""
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
            "difficulty": row["difficulty"]
        })
    return samples


def run_inference_round(samples, model_config, task, seed):
    """Run one round of inference and return predicted answers"""
    evaluator = Evaluator(
        task=task,
        config=EvalConfig(
            model=model_config.get('name', ''),
            temperature=1.0,
            # max_tokens=1024,
            max_tokens=4096,
            logprobs=False,
            seed=seed
        ),
        samples=copy.deepcopy(samples)
    )

    _ = evaluator.run_inference(
        format_fn=format_question_math,
        extract_fn=extract_answer,
        flag=False
    )

    preds = [s.get("PredAnswer", "").strip() for s in evaluator.samples]
    clear_mem()
    del evaluator
    return preds


def evaluate_accuracy(samples, preds):
    """Compute accuracy for a set of predictions"""
    accs = []
    for s, p in zip(samples, preds):
        gold = parse(s["answer"])
        ans = parse(p)
        accs.append(float(verify(gold, ans)))
    return accs


def run_evaluate_curve(
    task: str = "deepmath",
    model: str = "qwen-7b",
    data_path: str = "/mnt/shared-storage-user/yuzhiyin/RLEvol_new/deepmath_all_new.csv",
    num_infer: int = 3,
    output_dir: str = "/mnt/shared-storage-user/yuzhiyin/RLEvol_new/eval_curve"
):
    """
    Run multiple inference rounds and compute average accuracy per difficulty
    across all rounds (final average over n runs).
    """
    if task not in TASK_CONFIG:
        raise ValueError(f"Task {task} not found in config")
    if model not in MODELS_CONFIG:
        raise ValueError(f"Model {model} not found in config")

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔧 Using model: {model}")
    print(f"📘 Dataset: {data_path}")
    print(f"🔁 Total rounds: {num_infer}")

    model_config = configure_model(model)
    samples = load_data(data_path)

    # 用于汇总所有轮次结果
    all_round_accs = [[] for _ in range(len(samples))]
    all_round_stats = []

    for i in range(num_infer):
        print(f"\n🚀 Running round {i+1}/{num_infer}")
        seed = int(datetime.datetime.now().timestamp() * 1000) + i
        random.seed(seed)

        preds = run_inference_round(samples, model_config, task, seed)
        accs = evaluate_accuracy(samples, preds)

        # 累加每个样本的acc（用于之后平均）
        for idx, acc in enumerate(accs):
            all_round_accs[idx].append(acc)

        # 记录每轮的 per-difficulty 统计
        df = pd.DataFrame(samples)
        df["Accuracy"] = accs
        round_stats = df.groupby("difficulty")["Accuracy"].mean().reset_index()
        round_stats["Round"] = i + 1
        round_stats["OverallAcc"] = np.mean(accs)
        all_round_stats.append(round_stats)

        print(round_stats)

    # === 所有轮次整体平均 ===
    avg_accs = [np.mean(acc_list) for acc_list in all_round_accs]
    df_final = pd.DataFrame(samples)
    df_final["AvgAccuracy"] = avg_accs

    final_stats = df_final.groupby("difficulty")["AvgAccuracy"].mean().reset_index()
    final_stats["Round"] = "avg"
    final_stats["OverallAcc"] = np.mean(avg_accs)

    print("\n✅ Final Average Accuracy by Difficulty:")
    print(final_stats)

    # === 合并输出 ===
    stats_all = pd.concat(all_round_stats + [final_stats], ignore_index=True)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    save_path = os.path.join(output_dir, f"curve_result_avg_{model}_{timestamp}.csv")
    stats_all.to_csv(save_path, index=False)

    print(f"\n✅ Final results saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    fire.Fire(run_evaluate_curve)

