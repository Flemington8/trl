from datasets import Dataset
from trl.trainer import GRPOTrainer
from aiopslab.rewards import format_reward

conversation_0 = {
    "messages":[
    {
        "role": "user",
        "content": "No resources found in tidb-cluster namespace.\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get namespaces | grep tidb-cluster\")\n```"
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
]
}

conversation_1 = {
    "messages":[
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
        "content": "No resources found in tidb-cluster namespace.\n"
    },
    {
        "role": "assistant",
        "content": "1"
    },
]
}

conversations = [conversation_0, conversation_1]

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(conversations)

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=format_reward,
    train_dataset=dataset,
)

trainer.train()