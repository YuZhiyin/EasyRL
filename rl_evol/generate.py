# -*- coding: utf-8 -*-

import os
import copy
import pandas as pd
import numpy as np
import datetime
import random
from tqdm import tqdm

from EasyRL.RL.RLEvol.eval import Evaluator, EvalConfig
from EasyRL.RL.RLEvol.config import TASK_CONFIG, MODELS_CONFIG
from EasyRL.RL.RLEvol.common import *
from math_verify import parse, verify


def configure_model(model: str) -> dict:
    if model in MODELS_CONFIG:
        config = MODELS_CONFIG[model]
        if 'url' in config:
            os.environ['LLM_BASE_URL'] = config['url']
        if 'OPENAI_API_KEY' in config:
            os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
        return config
    return {'name': model}


def load_samples(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def run_three_inference(samples, model_config, task):
    all_predictions = []

    for i in range(3):
        seed = int(datetime.datetime.now().timestamp() * 1000) + i
        random.seed(seed)

        print(f"Running inference {i+1} with seed {seed}")

        evaluator = Evaluator(
            task=task,
            config=EvalConfig(
                model=model_config.get('name', ''),
                temperature=1.0,
                max_tokens=1024,
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

        all_predictions.append(evaluator.samples)

        del evaluator

    return all_predictions


def compute_pass_rate(original_samples, inference_list):
    num_samples = len(original_samples)

    for idx in range(num_samples):
        correct_count = 0

        for run_id in range(3):
            pred = inference_list[run_id][idx]['PredAnswer']

            if isinstance(pred, list):
                pred = str(pred[0])

            gold = parse(original_samples[idx]["answer"])
            answer = parse(pred)

            if verify(gold, answer):
                correct_count += 1

        pass_rate = correct_count / 3.0
        original_samples[idx]["pass_rate"] = pass_rate

    return original_samples


def main(
    task="deepmath",
    model="qwen-1.5b",
    input_csv="/mnt/shared-storage-user/yuzhiyin/RLEvol_new/save_deepmath/qwen1.5b-threthod0.4/unlabeled_all.csv",
    output_csv="unlabeled_sorted_by_passrate.csv"
):

    if task not in TASK_CONFIG:
        raise ValueError(f"Task {task} not found")

    if model not in MODELS_CONFIG:
        raise ValueError(f"Model {model} not found")

    model_config = configure_model(model)

    print("Loading data...")
    samples = load_samples(input_csv)

    print("Running 3x inference...")
    inference_list = run_three_inference(samples, model_config, task)

    print("Computing pass rate...")
    samples = compute_pass_rate(samples, inference_list)

    print("Sorting by pass_rate (descending)...")
    df = pd.DataFrame(samples)
    df = df.sort_values(by="pass_rate", ascending=False)

    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    main()
