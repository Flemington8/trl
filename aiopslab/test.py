from transformers import AutoTokenizer



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

flat = " ".join(f"[{m['role']}] {m['content']}" for m in conversation)

# 3. Tokenize
batch = tokenizer(
    [flat],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)

# 4. Access token IDs and tokens
input_ids = batch["input_ids"]             # Tensor of shape [1, seq_len]
attention_mask = batch["attention_mask"]

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("Tokens:", tokens)
print("Token IDs:", input_ids[0].tolist())
