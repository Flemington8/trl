import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from transformers import AutoTokenizer
from trl.data_utils import maybe_apply_chat_template


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# 2. Flatten conversation
conversations = {
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
        "content": "```\nkubectl get all -n tidb-cluster\n```"
    },
    {
        "role": "user",
        "content": "Error parsing response: No API call found!"
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
        "content": "```\nexec_shell(\"kubectl get tidbclusters -n tidb-cluster\")\n```"
    },
    {
        "role": "user",
        "content": "error: the server doesn't have a resource type \"tidbclusters\"\n"
    },
    {
        "role": "assistant",
        "content": "```\nexec_shell(\"kubectl get crd | grep tidb\")\n```"
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
        "content": "1"
    }
]
}
conversations_text = maybe_apply_chat_template(conversations, tokenizer)

print("\n=== Result ===")
print(conversations_text)
# === Result ===
# {'text': '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nNo resources found in tidb-cluster namespace.\n<|im_end|>\n<|im_start|>assistant\n```\nexec_shell("kubectl get pods -n tidb-cluster -o wide")\n```<|im_end|>\n<|im_start|>user\nNo resources found in tidb-cluster namespace.\n<|im_end|>\n<|im_start|>assistant\n```\nexec_shell("kubectl get namespaces | grep tidb-cluster")\n```<|im_end|>\n<|im_start|>user\ntidb-cluster             Active   20h\n<|im_end|>\n<|im_start|>assistant\n```\nkubectl get all -n tidb-cluster\n```<|im_end|>\n<|im_start|>user\nError parsing response: No API call found!<|im_end|>\n<|im_start|>assistant\n```\nexec_shell("kubectl get statefulsets -n tidb-cluster")\n```<|im_end|>\n<|im_start|>user\nNo resources found in tidb-cluster namespace.\n<|im_end|>\n<|im_start|>assistant\n```\nexec_shell("kubectl get tidbclusters -n tidb-cluster")\n```<|im_end|>\n<|im_start|>user\nerror: the server doesn\'t have a resource type "tidbclusters"\n<|im_end|>\n<|im_start|>assistant\n```\nexec_shell("kubectl get crd | grep tidb")\n```<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n```\nsubmit("Yes")\n```<|im_end|>\n<|im_start|>user\n1<|im_end|>\n'}