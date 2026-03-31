# 📦 ChineseDevBench

This is the project repository for **ChineseDevBench**, a developmental benchmark for evaluating language models against human linguistic behavior in Mandarin Chinese.

---

## Overview

ChineseDevBench evaluates language models by comparing their behavior with **child and adult linguistic data**.

It includes multiple tasks probing:

- compositional reasoning  
- word association structure  
- semantic fluency  
- age of acquisition  
- language production  
- semantic similarity  
- perceptual feature knowledge  

The benchmark is organized into two main components:

```
data/         # processed datasets for each task
evaluation/   # task-specific evaluation scripts
```

---

## 📂 Repository Structure

```
.
├── data/
│   ├── adult_asso
│   ├── adult_sem
│   ├── adult_sts
│   ├── child_acq
│   ├── child_ana
│   ├── child_asso
│   ├── child_flu
│   └── child_prod
│
├── evaluation/
│   ├── adult_asso
│   ├── adult_sem
│   ├── adult_sts
│   ├── child_acq
│   ├── child_ana
│   ├── child_asso
│   ├── child_flu
│   └── child_prod
```

Each folder corresponds to one task, with a mirrored structure between `data/` and `evaluation/`.

---

## Tasks

ChineseDevBench includes **8 tasks**, divided into child-oriented and adult-oriented groups.

### Child-Oriented Tasks

- **CHILD-ANA** — Compositional analogy  
- **CHILD-ASSOC** — Word association  
- **CHILD-FLU** — Verbal fluency  
- **CHILD-ACQ** — Word acquisition (AoA)  
- **CHILD-PROD** — Language production  

### Adult-Oriented Tasks

- **ADULT-STS** — Semantic textual similarity  
- **ADULT-ASSOC** — Word association norms  
- **ADULT-SEM** — Semantic feature prediction  

---

## ⚙️ Running Evaluation

Each task has its own evaluation script.

Example:

```bash
python evaluation/child_asso/eval_swow_rsa.py --model MODEL_NAME
```

Please refer to each task folder for task-specific arguments.

---

## 🤖 Model

We provide a trained **Child-GPT2 model**:

👉 Download here: https://huggingface.co/Eva1s/Child-GPT2_model

---

## 📥 Data Availability

Due to licensing constraints, some datasets must be downloaded separately:

- **CHILD-ACQ**
  - Wordbank: https://wordbank.stanford.edu/data/  
  - AoA norms Liu et al. (2011): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0016505 

- **CHILD-PROD**
  - CHILDES Mandarin: https://talkbank.org/childes/access/Chinese/
  - CPCSLD Feng, C., Wang, S., & Li, S. (2026).: https://pubmed.ncbi.nlm.nih.gov/41588154/ 

- **ADULT-ASSOC**
  - SWOW-ZH: https://smallworldofwords.org/en/project/research 

Instructions for downloading and preprocessing are provided in relevant task folders where applicable.

---

## 📖 Citation

If you use ChineseDevBench or its data in your research, please cite:

```bibtex
@misc{,
  title={},
  author={Your Name and Co-authors},
  year={2026}
}
```



---

# 📦 中文版

## 项目简介

ChineseDevBench 是一个用于评估语言模型与人类语言行为（儿童与成人）的发展型基准。

## 任务

包含 8 个任务：

### 儿童任务

- CHILD-ANA（类比推理）
- CHILD-ASSOC（词语联想）
- CHILD-FLU（语义流畅性）
- CHILD-ACQ（词习得年龄）
- CHILD-PROD（语言生成）

### 成人任务

- ADULT-STS（语义相似度）
- ADULT-ASSOC（词联想规范）
- ADULT-SEM（语义特征预测）

---

## ⚙️ 运行方式


```bash
python evaluation/child_asso/eval_swow_rsa.py --model MODEL_NAME
```



---

## 🤖 模型

我们提供训练好的 **Child-GPT2 模型**：

👉 下载方式: https://huggingface.co/Eva1s/Child-GPT2_model

---


## 数据说明

以下数据需自行下载：

  - Wordbank: https://wordbank.stanford.edu/data/  
  - AoA 数据 Liu et al. (2011) : https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0016505 

- **CHILD-PROD**
  - CHILDES 语料: https://talkbank.org/childes/access/Chinese/
  - CPCSLD 数据 Feng, C., Wang, S., & Li, S. (2026). : https://pubmed.ncbi.nlm.nih.gov/41588154/ 

- **ADULT-ASSOC**
  - SWOW-ZH: https://smallworldofwords.org/en/project/research 
---
## 📖 引用

如需在研究中使用Chinese DevBench或数据，请引用：

```bibtex
@article{,
  title={},
  author={Your Name and Co-authors},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
