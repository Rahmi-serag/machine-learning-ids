# Machine Learning Intrusion Detection System (IDS)

This repository contains my BSc thesis project: **Intrusion Detection System (IDS) using Machine Learning**.

The project implements a flow-based intrusion detection system that classifies network traffic as **normal** or **attack** using supervised machine learning.

## Project Overview

The system uses the CICIDS2017 dataset for training and evaluation. Network traffic is represented using flow-based features generated with CICFlowMeter. Several machine learning models were trained and compared, including:

- Decision Tree
- Random Forest
- Support Vector Machine
- Neural Network

Random Forest was selected as the final model based on the evaluation results.

## Main Features

- CICIDS2017 dataset preprocessing
- Binary classification: normal vs attack
- Model training and evaluation
- Custom normal and attack traffic collection in a Kali Linux and Ubuntu virtual lab
- PCAP-to-CSV conversion using CICFlowMeter
- Live detection pipeline for captured network traffic
- Docker support for reproducible Python environment

## Technologies Used

- Python
- scikit-learn
- pandas
- NumPy
- CICIDS2017
- CICFlowMeter
- Docker
- Kali Linux
- Ubuntu
- VirtualBox

## Repository Structure

```text
src/
  prepare_dataset.py
  prepare_custom_dataset.py
  train_ids.py
  train_ids_improved.py
  live_detector.py
  live_pipeline_improved.py

results/
  model_comparison_results.csv
  model_comparison_results_improved.csv
  training_feature_columns_improved.csv

thesis/
  Intrusion_Detection_System_IDS_using_Machine_Learning.pdf