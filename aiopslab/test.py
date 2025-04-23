from transformers import AutoTokenizer
from trl import apply_chat_template


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# 2. Flatten conversation
conversation = [
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get pods -n tidb-cluster -o wide\")\n```"
    },
    {
        "role": "env",
        "content": "No resources found in tidb-cluster namespace.\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get namespaces | grep tidb-cluster\")\n```"
    },
    {
        "role": "env",
        "content": "tidb-cluster             Active   20h\n"
    },
    {
        "role": "assistant",
        "content": "```\nkubectl get all -n tidb-cluster\n```"
    },
    {
        "role": "env",
        "content": "Error parsing response: No API call found!"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get statefulsets -n tidb-cluster\")\n```"
    },
    {
        "role": "env",
        "content": "No resources found in tidb-cluster namespace.\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get tidbclusters -n tidb-cluster\")\n```"
    },
    {
        "role": "env",
        "content": "error: the server doesn't have a resource type \"tidbclusters\"\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get crd | grep tidb\")\n```"
    },
    {
        "role": "env",
        "content": ""
    },
    {
        "role": "assistant",
        "content": "```\nsubmit(\"Yes\")\n```"
    },
    {
        "role": "env",
        "content": "1"
    }
]

result_prompt_completion = apply_chat_template(conversation, tokenizer)

print("\n=== RLHF Format (prompt-completion) ===")
print(f"Prompt: {result_prompt_completion['prompt']}")
print(f"Completion: {result_prompt_completion['completion']}")
