import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from transformers import AutoTokenizer
from trl.data_utils import apply_chat_template


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# 2. Flatten conversation example (`dict[str, list[dict[str, str]]`):
conversation = {
    "messages":[
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
}
result_messages = apply_chat_template(conversation, tokenizer)

print("\n=== Result ===")
print(result_messages)
