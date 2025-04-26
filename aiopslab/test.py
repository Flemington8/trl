import torch
from transformers import AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from trl.data_utils import maybe_apply_chat_template

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Example conversation (already in the right format)
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

conversations = [conversation_0, conversation_1]  # Wrap in a list to simulate multiple conversations

# Function to separate prompts and completions in multi-turn conversations
def prepare_aligned_multi_turn_masks(conversations, tokenizer, device="cpu"):
    """
    Generate masks for batched multi-turn conversations with position alignment.
    
    Handles conversations with different numbers of turns, ensuring that:
    - Masks for the nth prompt/completion align positionally across conversations
    - Conversations with fewer turns have proper masking for missing turns
    - logits_to_keep_mask correctly identifies only valid completion tokens
    """
    batch_size = len(conversations)
    
    # Step 1: Find the maximum number of turns across all conversations
    max_turns = max(len([m for m in conv["messages"] if m["role"] == "assistant"]) 
                   for conv in conversations)
    
    # Step 2: Process each conversation, separating by turns
    all_prompt_texts_by_turn = [[] for _ in range(max_turns)]
    all_completion_texts_by_turn = [[] for _ in range(max_turns)]
    
    # Track which conversations have which turns
    turn_presence = torch.zeros((batch_size, max_turns), dtype=torch.bool, device=device)
    
    for conv_idx, conversation in enumerate(conversations):
        messages = conversation["messages"]
        user_messages = [msg for msg in messages if msg["role"] == "user" or msg["role"] == "system"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        for turn_idx in range(min(len(assistant_messages), max_turns)):
            # Mark this turn as present
            turn_presence[conv_idx, turn_idx] = True
            
            # Current prompt for this turn
            prompt = [user_messages[turn_idx]] # list[dict[str, str]]

            # Current completion for this turn
            completion = [assistant_messages[turn_idx]]
            
            # Apply chat template
            prompt_dict = {"messages": [prompt[0]]}
            completion_dict = {"messages": [completion[0]]}
            
            prompt_text = maybe_apply_chat_template(prompt_dict, tokenizer)["text"]
            completion_text = maybe_apply_chat_template(completion_dict, tokenizer)["text"]
            
            # Add to the appropriate turn lists
            all_prompt_texts_by_turn[turn_idx].append(prompt_text) # all_prompt_texts_by_turn[turn_idx] -> list[str]
            all_completion_texts_by_turn[turn_idx].append(completion_text)
        
        # For remaining turns that this conversation doesn't have, add empty placeholders
        for turn_idx in range(len(assistant_messages), max_turns):
            # Add empty strings as placeholders
            all_prompt_texts_by_turn[turn_idx].append("")
            all_completion_texts_by_turn[turn_idx].append("")
    
    # Step 3: Process each turn separately and collect tensors
    prompt_ids_by_turn = []
    prompt_mask_by_turn = []
    completion_ids_by_turn = []
    completion_mask_by_turn = []
    
    for turn_idx in range(max_turns):
        # Tokenize all prompts and completions for this turn
        prompt_inputs = tokenizer(
            all_prompt_texts_by_turn[turn_idx],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        
        completion_inputs = tokenizer(
            all_completion_texts_by_turn[turn_idx],
            return_tensors="pt", 
            padding=True,
            padding_side="right",
            add_special_tokens=True,
        )
        
        # Store the IDs and masks
        prompt_ids_by_turn.append(prompt_inputs["input_ids"])
        prompt_mask_by_turn.append(prompt_inputs["attention_mask"])
        completion_ids_by_turn.append(completion_inputs["input_ids"])
        completion_mask_by_turn.append(completion_inputs["attention_mask"])
    
    # Step 4: Determine the sizes for full tensors
    turn_prompt_lengths = [ids.size(1) for ids in prompt_ids_by_turn]
    turn_completion_lengths = [ids.size(1) for ids in completion_ids_by_turn]
    
    total_prompt_length = sum(turn_prompt_lengths)
    total_completion_length = sum(turn_completion_lengths)
    total_length = total_prompt_length + total_completion_length
    
    # Step 5: Create full tensors with proper alignment
    full_ids = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    full_mask = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    prompt_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    completion_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    
    # Track positions for each turn
    current_pos = 0
    
    # Fill tensors for each turn
    for turn_idx in range(max_turns):
        p_length = turn_prompt_lengths[turn_idx]
        c_length = turn_completion_lengths[turn_idx]
        
        # Fill prompt section
        prompt_end_pos = current_pos + p_length
        full_ids[:, current_pos:prompt_end_pos] = prompt_ids_by_turn[turn_idx]
        full_mask[:, current_pos:prompt_end_pos] = prompt_mask_by_turn[turn_idx]
        
        # Set prompt mask for valid turns only
        for b in range(batch_size):
            if turn_presence[b, turn_idx]:
                prompt_mask[b, current_pos:prompt_end_pos] = prompt_mask_by_turn[turn_idx][b].bool()
        
        current_pos = prompt_end_pos
        
        # Fill completion section
        completion_end_pos = current_pos + c_length
        full_ids[:, current_pos:completion_end_pos] = completion_ids_by_turn[turn_idx]
        full_mask[:, current_pos:completion_end_pos] = completion_mask_by_turn[turn_idx]
        
        # Set completion mask for valid turns only
        for b in range(batch_size):
            if turn_presence[b, turn_idx]:
                completion_mask[b, current_pos:completion_end_pos] = completion_mask_by_turn[turn_idx][b].bool()
        
        current_pos = completion_end_pos
    
    # Step 6: Create the 1D logits_to_keep_mask (flattened version of completion_mask)
    # This identifies which positions should contribute to the loss calculation
    logits_to_keep_mask = torch.any(completion_mask, dim=0) # shape: (total_length,)
    
    return {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "prompt_mask": prompt_mask,
        "completion_mask": completion_mask,
        "logits_to_keep_mask": logits_to_keep_mask,
        "turn_presence": turn_presence
    }

# Test with our conversations
result = prepare_aligned_multi_turn_masks(conversations, tokenizer)

# Add visualizations to see the masks
def visualize_turn_alignment(result):
    """Visualize the turn alignment in the masks"""
    batch_size = result["prompt_mask"].shape[0]
    
    for i in range(batch_size):
        print(f"\nConversation {i}:")
        print("Prompt mask (P=token, .=masked):")
        p_mask = ''.join(['P' if t else '.' for t in result["prompt_mask"][i]])
        
        # Break into readable chunks
        for j in range(0, len(p_mask), 50):
            print(p_mask[j:j+50])
        
        print("\nCompletion mask (C=token, .=masked):")
        c_mask = ''.join(['C' if t else '.' for t in result["completion_mask"][i]])
        
        # Break into readable chunks
        for j in range(0, len(c_mask), 50):
            print(c_mask[j:j+50])
    
    print("\nTurn presence matrix:")
    for i in range(batch_size):
        present_turns = [str(j+1) for j in range(result["turn_presence"].shape[1]) 
                         if result["turn_presence"][i, j]]
        print(f"Conversation {i}: Turns {', '.join(present_turns)}")
    
    print("\nlogits_to_keep_mask statistics:")
    print(f"Total positions: {result['logits_to_keep_mask'].shape[0]}")
    print(f"Positions to keep: {result['logits_to_keep_mask'].sum().item()}")

# Run visualization
visualize_turn_alignment(result)