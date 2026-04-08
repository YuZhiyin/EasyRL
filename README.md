# EasyRL 

This repository provides the implementation of **EasyRL**, a data-efficient reinforcement learning framework for large language models (LLMs) that enables **self-evolution from easy labeled data to difficult unlabeled data**.

📄 **Paper:** *Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning*

---

## 🔥 Overview

EasyRL is inspired by cognitive learning theory and aims to answer:

> *Can LLMs evolve from limited easy labeled data to solve harder problems?*

The framework implements a semi-supervised RL training pipeline for reasoning through three stages:
1. **Knowledge Transfer:** Supervised GRPO training on labeled data.
2. **Divide-and-Conquer Pseudo Labeling:** Filtering and labeling on unlabeled data.
3. **Difficulty-Progressive Self-Training:** Supervised RL using generated pseudo-labels across multiple rounds.

---

## 🧩 Code Structure

* **`verl/`**: Core RL training codebase.
* **`rl_evol/`**: The pseudo-labeling and filtering pipeline for unlabeled data.
* **`eval_math/`**: Mathematical reasoning evaluation.
* **`eval_natural/`**: Natural language task evaluation.

---

## 🚀 Training Pipeline

### Step 1: Knowledge Transfer (Supervised GRPO)
Train a warm-up model ($M_{warm}$) using easy labeled data. 

```bash
bash EasyRL/verl/examples/ttrl/qwen2.5-1.5b.sh
```

**Reward Design:**
* **Correct answer:** +1
* **Incorrect answer:** 0
* **Format error:** -0.5

---

### Step 2: Pseudo Labeling + Selection
Run the self-evolving pipeline to generate high-quality training data from unlabeled samples ($D_{unlabeled}$).

```bash
python EasyRL/rl_evol/pipeline.py
```

This step implements a **Divide-and-Conquer** strategy:
1.  **Consistency-based Selection:** Performs multiple rollouts per sample and selects samples with identical outputs.
2.  **Reflection-based Resolution:** Computes entropy for inconsistent samples and applies a reflection mechanism to select low-uncertainty samples.

**Output:** Selected unlabeled dataset $D_{unlabeled\_selected}$.

---

### Step 3: Difficulty-Progressive Self-Training
Train the model using a combination of original labeled data and the new pseudo-labeled data.

* Combine $D_{label} + D_{unlabeled\_selected}$.
* Treat pseudo-labels as ground-truth supervision.
* Apply supervised RL (GRPO).

---

## 🔁 Iterative Self-Evolution

EasyRL is designed to be cyclic. Repeat the following loop to achieve an **Easy → Hard progression**:

This iterative process enables gradual performance improvement and autonomous self-evolution.

---

## 📊 Evaluation

To evaluate the model performance, navigate to the respective directories:

* **Math Tasks:** `cd eval_math/`
* **Natural Language Tasks:** `cd eval_natural/`

Please refer to the scripts within each directory for detailed usage instructions.

---

## 🙏 Acknowledgement

This repository is built upon the following works:
* [EMPO](https://github.com/QingyangZhang/EMPO)
* [SemiEvol](https://github.com/luo-junyu/SemiEvol)

We sincerely thank the authors for their valuable contributions to the field.