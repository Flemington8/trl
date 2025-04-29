# AIOpsLab

AIOpsLab is a framework for training and deploying AI models for ITOps and DevOps tasks using Group Relative Policy Optimization (GRPO). The framework specializes in training models to detect, localize, analyze, and mitigate system failures in Kubernetes and various infrastructure environments.

## Overview

This project leverages the [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) library to train language models for operational tasks. It uses GRPO, a reinforcement learning technique, to optimize model performance on specific AIOps tasks using reward functions.

## Project Structure

```
aiopslab/
├── __init__.py
├── client.py          # Client for interacting with the AIOps service
├── generator.py       # Conversation generator for the model
├── grpo.py            # Main GRPO training implementation
├── rewards.py         # Reward functions for training
├── test.py            # Test scripts
├── train_grpo.py      # Training script for GRPO
├── __pycache__/       # Python cache files
└── accelerate_configs/ # Configuration files for distributed training
    ├── deepspeed_zero1.yaml
    ├── deepspeed_zero2.yaml
    └── deepspeed_zero3.yaml
```

## Features

- Fine-tuning of language models for AIOps tasks using GRPO
- Support for multiple task types: detection, localization, analysis, and mitigation
- Integration with vLLM for efficient inference
- PEFT with LoRA for parameter-efficient fine-tuning
- DeepSpeed integration for distributed training
- Comprehensive dataset of AIOps scenarios

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aiopslab.git
cd aiopslab
```

2. Install the dependencies:
```bash
pip install -e .
```

3. Install TRL:
```bash
pip install trl
```

## Dependencies

- [TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [Accelerate](https://github.com/huggingface/accelerate)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [vLLM](https://github.com/vllm-project/vllm)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)

## Usage

### Starting the vLLM Server

Start the vLLM server to handle model inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-Coder-1.5B-Instruct --gpu_memory_utilization 0.5
```

### Training with GRPO

Run the GRPO training script with DeepSpeed:

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file aiopslab/accelerate_configs/deepspeed_zero2.yaml aiopslab/grpo.py
```

## Model Configuration

The project uses the Qwen2.5-Coder-0.5B-Instruct model with the following configurations:

- **PEFT Configuration**: LoRA with r=8, alpha=8, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj (attention modules)
- **Training Hyperparameters**: 
  - Batch size: 1 per device
  - Gradient accumulation steps: 8
  - Mixed precision: fp16
  - Number of generations: 4
  - Beta: 0.0

## Dataset

The dataset includes various AIOps tasks across different environments:

- Kubernetes misconfigurations
- MongoDB authentication issues
- Container and pod failures
- Network problems
- Service failures in various applications

Each problem is categorized by task type:
- **Detection**: Identifying if there's a problem
- **Localization**: Finding where the problem is
- **Analysis**: Understanding the problem
- **Mitigation**: Resolving the problem

## License

[Insert your license information here]

## Contributing

[Insert contribution guidelines here]

## Contact

[Insert contact information here]