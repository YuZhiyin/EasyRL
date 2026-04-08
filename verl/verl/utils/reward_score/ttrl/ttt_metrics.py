from collections import Counter
from typing import List
import math

from typing import List
import numpy as np
import random


from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify    

def binarize_max(frequencies):
    """
    把 frequencies 转换为一个 one-hot 向量，最大频率对应位置为 1，其余为 0。
    例子：[0.1, 0.4, 0.4, 0.1] → [0, 1, 1, 0]
    """
    arr = np.array(frequencies)
    max_val = np.max(arr)  # 获取最大值[7,9](@ref)
    return np.where(arr == max_val, 1, 0).tolist()  # 向量化替换

def semantic_cluster(task, model_answers, extra_info):
    representatives = []  # 存储独特的答案代表（包括每个空字符串）
    counts = []          # 存储每个独特答案对应的出现次数
    cluster_indices = []  # 存储每个答案对应的聚类索引

    n = len(model_answers)  # 总答案数量
    
    for i, ans in enumerate(model_answers):
        # 处理空字符串：每个空字符串都视为独特聚类
        if ans == "":
            representatives.append(ans)  # 添加空字符串作为新代表
            counts.append(1)              # 出现次数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新聚类索引
            continue
        
        # 处理非空字符串：尝试匹配已有聚类
        found = False
        for idx, rep in enumerate(representatives):
            # 跳过空字符串代表（避免非空字符串与空字符串比较）
            if rep == "":
                continue
            # 使用auto_verify判断答案是否匹配
            if auto_verify(task, [ans], [rep], extra_info=extra_info)[0][0]:
                counts[idx] += 1          # 增加计数
                cluster_indices.append(idx)  # 记录聚类索引
                found = True
                break
        
        # 未找到匹配则创建新聚类
        if not found:
            representatives.append(ans)  # 添加新代表
            counts.append(1)              # 新聚类计数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新索引

    # 计算每个答案的频率（长度为n的列表）
    frequencies = [counts[idx] / n for idx in cluster_indices]
    
    # 计算每个独特答案的频率（长度为len(representatives)的列表）
    unique_frequencies = [c / n for c in counts]
    
    # 返回：每个答案的频率列表，所有独特答案列表，独特答案频率列表
    return frequencies, representatives, unique_frequencies

def entropy_thresholding(frequencies, unique_frequencies, low, high):
    n = len(unique_frequencies)

    entropy = 0.0
    for p in unique_frequencies:
        if p > 0:
            entropy -= p * math.log(p)
    
    max_entropy = math.log(n)
    
    min_valid = low * max_entropy
    max_valid = high * max_entropy
    
    if entropy >= min_valid and entropy <= max_valid:
        return frequencies
    else:
        return [0.0] * len(frequencies)


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    entropys: List = None,
    task="math", extra_info=None, reward_type=None, entropy_thres=None):
    
    print("entropys in test_time_train_metrics: ", entropys)
    
    difficulty = 0
    w_int = 1
    
    print("extra info",extra_info)

    if reward_type == 'semi':
        if isinstance(extra_info, dict):
            difficulty = extra_info.get("difficulty", 3.0)
        elif isinstance(extra_info, list) and len(extra_info) > 0:
            difficulty = extra_info[0].get("difficulty", 3.0) if isinstance(extra_info[0], dict) else extra_info[0]
        else:
            difficulty = 3.0
            
        print(difficulty)

    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    frequencies, unique_answers, unique_frequencies = semantic_cluster(task, model_answers, extra_info)
    assert reward_type in ['gt', 'semantic_entropy', 'voting', 'semi']
    if reward_type == 'voting':
        # TTRL rewards
        rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    elif reward_type == 'semantic_entropy':
        # EMPO rewards
        rewards = entropy_thresholding(frequencies, unique_frequencies, low=0.0, high=entropy_thres)
    elif reward_type == 'gt':
        # true rewards
        rewards = true_rewards
    elif reward_type == 'semi':
        ##外部伪标注奖励
        r_extrinsic = np.array(true_rewards) * math.log(difficulty - 1) 
        ##模型内部奖励       
        r_intrinsic = np.array(binarize_max(frequencies), dtype=float)
        # r_intrinsic = np.array(frequencies)
        rewards_raw = r_extrinsic + w_int * r_intrinsic
        rewards = rewards_raw.tolist()
    
        print("r_extrinsic:", r_extrinsic.tolist())
        print("r_intrinsic:", r_intrinsic.tolist())
        print("rewards:", rewards)


    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate, ## 多数投票答案是否等于 ground truth
        "reward_accuracy": rewards_hit_rate, ## 计算 reward_type 得到的 reward 和 ground truth reward 的 相等比例
        "majority_ratio": majority_ratio, ## 最多答案的比例，衡量模型输出一致性
        "mean_train_accuracy": sum(true_rewards) / len(true_rewards), ## 模型输出整体正确率（基于 ground-truth）
        "mean_reward": sum(rewards) / len(rewards), ##当前 reward 的平均值
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0, ## N个中是否至少有1个是正确的
    }
    print("ttrl_metrics: ",ttrl_metrics)
    return rewards, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_mean_train_accuracy": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics

# def post_semantic_cluster(model_answers, equality):
#     # 存储每个答案的聚类索引（初始为-1）
#     cluster_indices = [-1] * len(model_answers)
#     # 存储每个聚类的频率（长度等于聚类数量）
#     cluster_counts = []
#     # 存储每个聚类的代表（第一个加入该聚类的答案）
#     representatives = []
    
#     n = len(model_answers)
    
#     # 遍历每个答案
#     for i, ans in enumerate(model_answers):
#         if ans == "":
#             # 对于空字符串，直接创建新聚类
#             cluster_indices[i] = len(cluster_counts)
#             cluster_counts.append(1)
#             representatives.append(ans)
#             continue
            
#         # 尝试加入已有聚类（只检查非空聚类）
#         found = False
#         for j in range(len(cluster_counts)):
#             if representatives[j] == "":
#                 continue  # 跳过空聚类
#             rep_index = cluster_indices.index(j)  # 找到该聚类的第一个成员索引
#             if equality[i][rep_index]:
#                 cluster_indices[i] = j
#                 cluster_counts[j] += 1
#                 found = True
#                 break
                
#         # 未找到匹配则创建新聚类
#         if not found:
#             cluster_indices[i] = len(cluster_counts)
#             cluster_counts.append(1)
#             representatives.append(ans)
    
#     # 计算每个答案的频率（长度等于答案数量）
#     frequencies = [cluster_counts[idx] / n for idx in cluster_indices]
#     # 计算每个聚类的频率（长度等于聚类数量）
#     unique_frequencies = [count / n for count in cluster_counts]
    
#     return frequencies, representatives, unique_frequencies

# def run_sequential_auto_verify(
#     auto_verify_func,
#     model_answers: list[str],
#     task_str: str,
#     extra_info=None,
#     # timeout and num_workers are not used but kept for signature compatibility
#     timeout: int = 60,
#     num_workers: int = 1
# ) -> list[list[int]]:
#     """
#     Performs sequential (single-process) auto-verification. Useful for debugging.
#     """
#     N = len(model_answers)
#     result_matrix = [[0] * N for _ in range(N)]

#     # Use tqdm for a progress bar, as this can be slow
#     # pbar = tqdm(total=(N * (N - 1) // 2), desc="Sequential Auto-Verification")
    
#     for i in range(N):
#         result_matrix[i][i] = 1 # An answer is always equivalent to itself
#         for j in range(i + 1, N):
#             try:
#                 # Directly call the function
#                 result = auto_verify_func(task_str, [model_answers[i]], [model_answers[j]], extra_info)
                
#                 # Handle both tuple and int return types
#                 value = result[0][0] if isinstance(result, tuple) else result
#                 result_matrix[i][j] = value
#                 result_matrix[j][i] = value # Similarity Matrix is symmetric

#             except Exception:
#                 e = traceback.format_exc()
#                 print(f"\n[Error] Verification for pair ({i}, {j}) failed: {e}")
#                 result_matrix[i][j] = 0
#                 result_matrix[j][i] = 0
#             # finally:
#             #    pbar.update(1)

#     # pbar.close()
#     # print("\n[Success] All sequential auto-verification tasks are complete.")
#     return result_matrix

# def test_time_train_metrics(
#     solutions: List[str],
#     ground_truth: List[str],
#     task="math", extra_info=None, reward_type=None, entropy_thres=None):
#     assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    
#     difficulty = 0
#     w_int = 1.0
    
#     print("extra info",extra_info)

#     if reward_type == 'semi':
#         if isinstance(extra_info, dict):
#             difficulty = extra_info.get("difficulty", 3.0)
#         elif isinstance(extra_info, list) and len(extra_info) > 0:
#             difficulty = extra_info[0].get("difficulty", 3.0) if isinstance(extra_info[0], dict) else extra_info[0]
#         else:
#             difficulty = 3.0

#     if isinstance(ground_truth[0], list):
#         ground_truth = [gt[-1] for gt in ground_truth]
#     assert (
#         isinstance(ground_truth, list) and
#         all(isinstance(item, str) for item in ground_truth)
#     ), "Ground truth must be list[str]"
    
#     assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {set(ground_truth)}"
#     ground_truth_str = ground_truth[0]
    
#     model_answers = auto_extract(task, solutions, extra_info=extra_info)
#     assert (
#         isinstance(model_answers, list) and
#         all(isinstance(item, str) for item in model_answers)
#     ), "Model answers must be list[str]"

#     answers_with_gt = model_answers + [ground_truth_str]
#     result_matrix = run_sequential_auto_verify(auto_verify, answers_with_gt, task, extra_info=extra_info)
#     equality_matrix = [row[:-1] for row in result_matrix[:-1]]
#     true_rewards = result_matrix[-1][:-1]
#     assert len(true_rewards) == len(model_answers)

#     frequencies, unique_answers, unique_frequencies = post_semantic_cluster(model_answers, equality_matrix)
    
#     # Handle case where frequencies is empty
#     if not frequencies:
#         hit_rate = 0.0
#         majority_ratio = 0.0
#         estimated_label = ""
#     else:
#         max_index = np.argmax(frequencies)
#         estimated_label = model_answers[max_index]
#         majority_ratio = frequencies[max_index]
#         hit_rate = float(true_rewards[max_index])

#     assert reward_type in ['gt', 'entropy', 'voting', 'format', 'random', 'best', 'semi']
#     if reward_type == 'voting':
#         rewards = binarize_max(frequencies)
#     elif reward_type == 'entropy':
#         rewards = frequencies
#     elif reward_type == 'gt':
#         rewards = true_rewards
#     elif reward_type == 'best':
#         rewards = true_rewards if hit_rate < 1 else [0.0] * len(true_rewards)
#     elif reward_type == 'random':
#         rewards = [random.uniform(0, 1) for _ in range(len(true_rewards))]
#     elif reward_type == 'format':
#         rewards = [1.0 if len(ans) > 0 else 0 for ans in model_answers]
#     elif reward_type == 'semi':
#         ##外部伪标注奖励
#         r_extrinsic = np.array(true_rewards) * math.log(difficulty - 1)
#         ##模型内部奖励
#         # H_list = np.array([compute_token_entropy(ans) for ans in model_answers])
#         # H_max = max(H_list.max(), 1e-6)
#         # r_intrinsic = 1.0 - H_list / H_max  
        
#         r_intrinsic = np.array(binarize_max(frequencies), dtype=float)
#         ##总奖励归一化
#         rewards_raw = r_extrinsic + w_int * r_intrinsic
        
#         rewards = rewards_raw.tolist()
        
#         # rewards_min, rewards_max = rewards_raw.min(), rewards_raw.max()
#         # if rewards_max > rewards_min:
#         #     rewards = (rewards_raw - rewards_min) / (rewards_max - rewards_min)
#         # else:
#         #     rewards = np.zeros_like(rewards_raw)
#         # rewards = rewards.tolist()
        
#         print("r_extrinsic:", r_extrinsic.tolist())
#         print("r_intrinsic:", r_intrinsic.tolist())
#         print("rewards:", rewards)
    
#     rewards_hit_rate = sum(1 for r, tr in zip(rewards, true_rewards) if r == tr) / len(rewards) if rewards else 0

#     assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

#     metrics = {
#         "label_accuracy": [hit_rate] * len(model_answers),
#         "reward_accuracy": [rewards_hit_rate] * len(model_answers),
#         "majority_ratio": [majority_ratio] * len(model_answers),
#         "train_accuracy": [sum(true_rewards) / len(true_rewards) if true_rewards else 0.0] * len(model_answers),
#         "train_reward": [sum(rewards) / len(rewards) if rewards else 0.0] * len(model_answers),
#         f"pass@{len(solutions)}": [1.0 if sum(true_rewards) >= 1 else 0.0] * len(model_answers),
#         "extracted_answers": model_answers,
#         "estimated_label": [estimated_label] * len(model_answers),
#         # "filtered": [filtered] * len(model_answers),
#         # "reward_noise": [reward - true_reward for reward, true_reward in zip(rewards, true_rewards)]
#     }
    
#     print(metrics)

#     return rewards, metrics