# ECE226 Final Project: Test-Time Scaling for Small Language Models

**UCSD ECE 226 — Winter 2026**

## Overview

This project investigates whether small language models (0.8B–4B parameters) can recover accuracy through **Best-of-N self-consistency decoding** at inference time, without any retraining or architectural changes. We evaluate multiple open-weight models on GSM8K (mathematical reasoning) and ARC-Challenge (commonsense reasoning), systematically characterizing how accuracy scales with the number of inference samples N and analyzing the accuracy-latency tradeoff.

## Repository Files

- `qwen_scratchpad.ipynb`: contains quick experiments as a proof of concept during experimental design. Runs the self-consistency paradigm on QWEN and GSM8K.
- `test_time_scaling_qwen3_5_0_8b.ipynb`: contains experiments for running QWEN3.5-0.8B on GSM8K. Can be adapted to run different models on different dataset (ARC) by changing the model_name and dataset variable.
