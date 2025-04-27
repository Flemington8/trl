from datasets import Dataset
from trl.trainer import GRPOTrainer
from trl.trainer import GRPOConfig
from peft import LoraConfig
from aiopslab.rewards import format_reward

conversation_0 = {
    "messages":[
    {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "user",
        "content": "tidb-cluster             Active   20h\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
    {
        "role": "user",
        "content": ""
    },
    {
        "role": "assistant",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
        {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
        {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
        {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
]
}

conversation_1 = {
    "messages":[
    {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
    {
        "role": "user",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "assistant",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
]
}

conversations = [conversation_0] + [conversation_1 for _ in range(100)]

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(conversations)

training_args = GRPOConfig(output_dir="./output/Qwen2.5-Coder-0.5B-Instruct-GRPO",
                        beta=0.0,
                        fp16=True,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=8,
                        report_to=[])

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,            # Scaling factor for LoRA
    lora_dropout=0.05,
    bias="none",              # Don't train bias parameters to save memory
    task_type="CAUSAL_LM",    # Task type for the model
    # Target specific attention modules in Qwen2.5 architecture
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ],
    # Additional memory-saving options
    inference_mode=False,     # We're training, not inferencing
    modules_to_save=[],       # Don't fully save any modules
)

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    reward_funcs=format_reward,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    is_conversation=True,
)

trainer.train()