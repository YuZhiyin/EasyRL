from EasyRL.RL.RLEvol.common import *

MODELS_CONFIG = {
    "qwen-7b":{
        "name": "/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen2.5-Math-7B",
        "adapter": ""
    },
    "qwen-1.5b":{
        "name": "/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen2.5-Math-1.5B",
        "adapter": ""
    }
}


TASK_CONFIG = {
    "deepmath": {
        "dataset_name": "deepmath",
        "test_path": "./data/deepmath/test.csv",
        "labeled_path": "./data/deepmath/labeled.csv",
        "unlabeled_path": "./data/deepmath/unlabeled.csv",
        "question_type": "math",
        "additional_prompt": "Let's think step by step and output the final answer within \\boxed{}.",
        "check_fn": exact_match
    }
}

