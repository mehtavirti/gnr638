# GNR638 Assignment – Model Evaluation Guide

This repository contains the codebase for building the C++ backend and evaluating two models on a given dataset.

Follow the steps below to set up the environment and run the evaluation scripts.

---

## 📦 Setup Instructions

### 1. Download and Extract

- Download the ZIP folder.
- Extract it to your desired location.

---

### 2. Open Command Prompt / Terminal

Navigate to the extracted folder:

```bash
cd gnr638-master

cd framework/cpp
py setup.py build_ext --inplace
cd ../..
```
To test model 1: run 
```bash
py model1_eval.py {datasetpath} outputs/model_weights.txt
```
To test model 2: run
```bash
py model2_eval.py {dataset} best_model.pkl
```
