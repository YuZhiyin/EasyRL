# -*- coding: utf-8 -*-

import os
import fire
import pandas as pd
import random
import copy
import numpy as np
from EasyRL.RL.RLEvol.train_sft import train_sft
from EasyRL.RL.RLEvol.common import *
from EasyRL.RL.RLEvol.eval import Evaluator, EvalConfig
from EasyRL.RL.RLEvol.config import TASK_CONFIG, MODELS_CONFIG
import datetime
from tqdm import tqdm
from math_verify import parse, verify

def calculate_entropy(probs): ##计算给定预测概率的熵（不确定性指标）
    prob_list = np.array(probs)
    entropy = - np.sum(prob_list) / len(prob_list)
    return entropy

def configure_model(model: str) -> dict:
    """Configure model settings"""
    if model in MODELS_CONFIG:
        config = MODELS_CONFIG[model]
        if 'url' in config:
            os.environ['LLM_BASE_URL'] = config['url']
        if 'OPENAI_API_KEY' in config:
            os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
        return config
    return {
        'name': model
    }

def load_samples(data_path: str, task_config: dict) -> list:
    ## 读取 .csv 格式的数据并添加任务配置信息，如问题类型、额外提示
    df = pd.read_csv(data_path)
    samples = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        samples.append(sample)
    return samples

def prepare_data(task, task_config, labeled_path=None, unlabeled_path=None, output_dir=None):
    ##准备未标注数据
    """Prepare and embed data for inference"""
    if not unlabeled_path:
        unlabeled_path = f'data/{task}/unlabeled.csv'
    
    unlabel_data = load_samples(unlabeled_path, task_config)
    
    return unlabel_data

def run_multiple_inference(all_data: list, num_infer: int, model_config: dict, task: str, base_adapter: str = None) -> list:
    ##多次调用模型推理（num_infer次）
    """Run multiple inferences with progress tracking"""
    all_predictions = []
    
    for i in tqdm(range(num_infer), desc="Running inferences"):
        current_seed = int(datetime.datetime.now().timestamp() * 1000) + i
        random.seed(current_seed)
        
        print(f"Running inference {i+1} with seed {current_seed}")

        eval_instance = Evaluator(
            task=task,
            config=EvalConfig(
                model=model_config.get('name', ''),
                temperature=1.0,
                max_tokens=1024,
                logprobs=True,
                # lora_path=base_adapter,
                seed=current_seed
            ),
            samples=copy.deepcopy(all_data)
        )
        
        _ = eval_instance.run_inference(
            format_fn=format_question_math,
            extract_fn=extract_answer,
            flag=False
        )
        all_predictions.append(eval_instance.samples)

        del eval_instance
        clear_mem()
    
    return all_predictions

def process_results(unlabel_data, inference_list, num_infer=4):
    """
    将 num_infer 次的推理结果合并、对比并分析一致性：
    如果多次推理结果相同，则被认为是“一致样本（confident）”，可以直接当伪标签。
    如果有差异，则归为“不一致（uncertain）”，可能需要额外处理。
    """
    """Process inference results to generate pseudo-labels"""
    save_data = copy.deepcopy(unlabel_data)
    num_examples = len(unlabel_data)
    
    conf_samples = []
    unconf_samples = []
    consistent_indices = []

    for idx in range(num_examples):
        pred_list = []
        for i in range(num_infer):
            pred = inference_list[i][idx]['PredAnswer']
            if type(pred) == list:
                pred = str(pred[0])
            pred_list.append(pred)
        # print("第",idx+1,"题推理的pred_list是: ",pred_list)

        # entropy = calculate_entropy(inference_list[0][idx]['logprobs'])

        is_all_equal = True
        for i in range(1, len(pred_list)):
            gold = parse(pred_list[0])
            answer = parse(pred_list[i])
            result = verify(gold , answer)
            if not result:
                is_all_equal = False
                break

        if not is_all_equal:
            save_data[idx]['consist'] = 0
            save_data[idx]['entropy'] = None
            save_data[idx]['PredAnswers'] = pred_list
            save_data[idx]['Preds'] = [inference_list[i][idx]['Pred'] for i in range(num_infer)]
            unconf_samples.append(save_data[idx])
        else:  # 一致，生成伪标签（PseudoLabel）
            save_data[idx]['PseudoLabel'] = pred_list[0]
            save_data[idx]['consist'] = 1
            save_data[idx]['entropy'] = None
            conf_samples.append(save_data[idx])
            consistent_indices.append(idx)
    
    print(f'Consistent Rate: {len(conf_samples) / num_examples:.4f}')
    print(f'Inconsistent Rate: {len(unconf_samples) / num_examples:.4f}')
    
    # print(f'Consistent sample indices: {[i + 1 for i in consistent_indices]}')

    return conf_samples, unconf_samples

def resolve_inconsistencies(unconf_samples, model_config, task, base_adapter=None):
## 对“预测不一致的样本”再次运行推理，这次使用反思式提示（format_reflection）：
## 得到新的预测结果并计算熵。
## 仅保留熵低于一定阈值的（更确定的）样本作为“可接受伪标签”
    """Resolve inconsistent predictions with additional inference"""
    if not unconf_samples:
        return []

    eval_instance = Evaluator(
            task=task,
            config=EvalConfig(
                model=model_config.get('name', ''),
                temperature=1.0,
                max_tokens=4096,
                logprobs=True
                # lora_path=base_adapter,
        ),
        samples=copy.deepcopy(unconf_samples)
    )

    _ = eval_instance.run_inference(
        format_fn=format_reflection_math,
        extract_fn=extract_answer,
        flag=True
    )

    unconsis_preds = eval_instance.samples

    for i in range(len(unconsis_preds)):
        pred = unconsis_preds[i].get("PredAnswer", "").strip()

        # 如果 PredAnswer 为空，则尝试从 PredAnswers 中选第一个非空的作为 PseudoLabel
        if not pred:
            pred_list = unconsis_preds[i].get("PredAnswers", [])
            non_empty = next((p for p in pred_list if p.strip()), "")
            unconsis_preds[i]["PseudoLabel"] = non_empty
        else:
            unconsis_preds[i]["PseudoLabel"] = pred

        unconsis_preds[i]["entropy"] = calculate_entropy(unconsis_preds[i]["logprobs"])


    entropy_values = [s['entropy'] for s in unconsis_preds]
    entropy_threshold = np.percentile(entropy_values, 30)
    resolved_samples = [s for s in unconsis_preds if s['entropy'] < entropy_threshold]

    return resolved_samples

def calculate_accuracy(save_data):
    overall_scores = []
    """Calculate accuracy of pseudo-labels"""
    for i, s in enumerate(save_data):
        if "answer" in s and "PseudoLabel" in s:
            gold = parse(s["answer"])
            answer = parse(s["PseudoLabel"])
            is_correct = verify(gold, answer)
            score = float(is_correct)
            save_data[i]["Accuracy"] = score
            overall_scores.append(score)

    overall_accuracy = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    total = len(overall_scores)
    
    print(f"📊 共计样本数: {total}")
    print(f"✅ Overall Accuracy: {overall_accuracy:.4f}")
    return save_data

def save_results(save_data, task, model, output_dir=None):
## 将处理后的伪标签结果保存为CSV文件，便于后续使用或分析
    """Save results to CSV"""
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    if not output_dir:
        output_dir = f"save/{model}"
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{task}_pseudo_{timestamp}.csv")
    
    save_df = pd.DataFrame(save_data)
    save_df.to_csv(save_path, index=False, escapechar='\\')
    print(f"Saved pseudo-labels to {save_path}")
    
    return save_path

def run_pipeline(
    task: str = "deepmath",
    model: str = "qwen-1.5b",
    num_infer: int = 2,
    base_model_path="/fs-computility/MA4Tool/shared/MA4Tool/hug_ckpts/Qwen2.5-Math-1.5B",
    output_dir: str = "/mnt/shared-storage-user/yuzhiyin/RLEvol_new/save_deepmath/qwen1.5b-round1",
    labeled_path: str = None,
    unlabeled_path: str ="/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/code/EasyRL/RLEvol_new/data/deepmath/unlabeled_all.csv"
):
    """
    Run the complete SemiEvol pipeline
    
    Args:
        task: Task name (e.g. 'mmlu', 'arc')
        model: Model name from config
        num_infer: Number of inference iterations
        base_model_path: Path to base model (optional)
        output_dir: Directory to save results (optional)
        labeled_path: Path to labeled data (optional)
        unlabeled_path: Path to unlabeled data (optional)
    """
    
    if task not in TASK_CONFIG:
        raise ValueError(f"Task {task} not found in config")
    
    print("task: ",task)
    
    if model not in MODELS_CONFIG:
        raise ValueError(f"Model {model} not found in config")
    
    print("model: ",model)
 
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        output_dir = f"save/{model}/{task}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the environment
    task_config = TASK_CONFIG[task]
    model_config = configure_model(model)
    model_path = base_model_path if base_model_path else model_config['name']
    
    print("model path: ",model_path)
  
    print("\n=== Step 1: Prepare Data ===")

    unlabel_data = prepare_data(task, task_config, labeled_path, unlabeled_path, output_dir)
    print("successfully prepare data!!!")
    
    # Step 2: Run multiple inferences
    print(f"\n=== Step 2: Running Inference ({num_infer} times) ===")
    inference_list = run_multiple_inference(unlabel_data, num_infer, model_config, task)
    
    # Step 3: Process results
    print("\n=== Step 3: Processing Inference Results ===", flush=True)
    conf_samples, unconf_samples = process_results(unlabel_data, inference_list, num_infer)
    
    # Step 4: Resolve inconsistencies
    print("\n=== Step 4: Resolving Inconsistent Predictions ===")
    resolved_samples = resolve_inconsistencies(
        unconf_samples, model_config, task
    )
    
    all_pseudo_labeled_data = conf_samples + resolved_samples
    
    # 保存最终结果
    print("\n=== Step 5: Saving Pseudo-labeled Results ===")
    # 如果需要，可以计算伪标签的准确率（如果unlabeled.csv有真实答案'answer'列）
    all_pseudo_labeled_data = calculate_accuracy(all_pseudo_labeled_data)
    save_path = save_results(all_pseudo_labeled_data, task, model, output_dir)
    print(f"Pipeline finished. All pseudo-labeled data saved to: {save_path}")
    
if __name__ == "__main__":
    fire.Fire(run_pipeline)
