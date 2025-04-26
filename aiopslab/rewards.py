import re
from typing import Callable, Dict, Optional

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

completions = [
    [{"role": "assistant", "content": "```\nexec_shell(\"kubectl get pods -n test-social-network | grep compost-post-service\")\n```"}], # Correct format
    [{"role": "assistant", "content": "Action: exec_shell(\"kubectl get configmaps -n test-social-network\")"}], # Incorrect format
    [{"role": "assistant", "content": "exec_shell(\"kubectl get pods -n test-social-network\")"}], # Incorrect format
    [{"role": "assistant", "content": "```\nexec_shell(\"kubectl get pods\")\n```"}], # Correct format
    [{"role": "assistant", "content": "submit([\"compost-post-service\", \"compost-post-service-1\", \"compost-post-service-2\"])\n```"}], # Incorrect format
    [{"role": "assistant", "content": "```\nsubmit(\"Yes\")\n```"}], # Correct format for detection
    [{"role": "assistant", "content": "```\nsubmit([\"TiDB Operator not installed\", \"Missing CRD for tidbclusters\", \"tidb-cluster namespace has no running pods\"])\n```"}], # Correct format for localization
    [{"role": "assistant", "content": "```\nsubmit({'system_level': 'Virtualization', 'fault_type': 'Misconfiguration'})\n```"}], # Correct format for root cause analysis
    [{"role": "assistant", "content": "```\nsubmit()\n```"}], # Correct format for mitigation
]

results = [
     {  "task": "detection",
        "Detection Accuracy": "Correct",
        "TTD": 105.1446475982666,
        "steps": 4,
        "in_tokens": 10152,
        "out_tokens": 73
    },
    {   "task": "localization",
        "Localization Accuracy": 0.0,
        "TTL": 20.57143259048462,
        "steps": 5,
        "in_tokens": 2187,
        "out_tokens": 140,
        "success": False,
        "is_subset": False
    },
    {   "task": "analysis",
        "system_level_correct": True,
        "fault_type_correct": False,
        "success": False,
        "TTR": 21.355182647705078,
        "steps": 9,
        "in_tokens": 5943,
        "out_tokens": 390
    },
    {   "task": "mitigation",
        "TTM": 14.532860279083252,
        "steps": 4,
        "in_tokens": 688,
        "out_tokens": 91,
        "success": False
    }
]

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has the correct API call format."""
#     # Check if reasoning model parameter is passed
#     is_reasoning = kwargs.get("is_reasoning", False)

#     if is_reasoning:
#         # Standard pattern for reasoning models (no need for think/answer format)
#         pattern = r"^```\nexec_shell\([\"].*?[\"].*\)\n```$|^```\nsubmit\(.*\)\n```$"
#     else:
#         # Pattern requiring think/answer format for non-reasoning models
#         pattern = r"^<think>.*?</think><answer>```\nexec_shell\([\"].*?[\"].*\)\n```</answer>$|^<think>.*?</think><answer>```\nsubmit\(.*\)\n```</answer>$"
    
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [0.0 if match else -1.0 for match in matches]

def format_reward(conversations, **kwargs):
    """
    Reward function that evaluates conversations based on the format correctness
    of completions.
    
    Args:
        conversations: List of conversation dictionaries, each containing a "messages" list
        **kwargs: Additional parameters, including is_reasoning flag
        
    Returns:
        List of reward scores, one per conversation
    """
    # Check if reasoning model parameter is passed
    is_reasoning = kwargs.get("is_reasoning", False)

    # Define the pattern based on reasoning mode
    if is_reasoning:
        # Standard pattern for reasoning models
        pattern = r"^```\nexec_shell\([\"].*?[\"].*\)\n```$|^```\nsubmit\(.*\)\n```$"
    else:
        # Pattern requiring think/answer format for non-reasoning models
        pattern = r"^<think>.*?</think><answer>```\nexec_shell\([\"].*?[\"].*\)\n```</answer>$|^<think>.*?</think><answer>```\nsubmit\(.*\)\n```</answer>$"
    
    # Calculate reward for each conversation
    conversation_rewards = []
    
    for conversation in conversations:
        # Extract completions from the conversation
        completions = [msg for msg in conversation["messages"] 
                             if msg.get("role") == "assistant"]
        
        # Check format of each completion
        format_checks = [bool(re.match(pattern, completion["content"])) for completion in completions]
        
        # Calculate conversation-level metrics
        total_completions = len(format_checks)
        correct_completions = sum(format_checks)
        
        # Calculate correctness ratio (0.0 to 1.0)
        correctness_ratio = correct_completions / total_completions
        
        # Convert to reward score (-1.0 to 1.0)
        # Perfect format: 1.0, All wrong: -1.0
        reward = (correctness_ratio * 2) - 1.0
        
        conversation_rewards.append(reward)
    
    return conversation_rewards

def detection_eval(result):
    # Check correctness of detection
    if result["Detection Accuracy"] == "Correct":
        detection_correct = True
    else:
        detection_correct = False

    # Check the detection time (TTD) is within acceptable limits
    TTD_threshold = 100  # Define threshold for fast detection
    TTD_penalty = 0.0
    if result["TTD"] > TTD_threshold:
        TTD_penalty = -0.5  # Penalty for long detection time

    # If detection is correct, reward, else penalize
    reward = 1.0 if detection_correct else -1.0
    return reward + TTD_penalty

def localization_eval(result):
    # Check the localization accuracy
    if result["Localization Accuracy"] > 0.5:  # Customize this threshold based on your task's requirement
        localization_correct = True
    else:
        localization_correct = False

    # Check the localization time (TTL) is within acceptable limits
    TTL_threshold = 20  # Define threshold for fast localization
    TTL_penalty = 0.0
    if result["TTL"] > TTL_threshold:
        TTL_penalty = -0.5  # Penalty for long localization time

    # If localization is correct, reward, else penalize
    reward = 1.0 if localization_correct else -1.0
    return reward + TTL_penalty

def analysis_eval(result):
    # Check if system-level analysis is correct
    system_level_correct = result["system_level_correct"]
    fault_type_correct = result["fault_type_correct"]
    
    # Penalty for incorrect fault type
    penalty = 0.0
    if not fault_type_correct:
        penalty = -0.5  # Penalty for incorrect fault type

    # Check Time to Resolution (TTR) for speed
    TTR_threshold = 30  # Define threshold for fast analysis
    TTR_penalty = 0.0
    if result["TTR"] > TTR_threshold:
        TTR_penalty = -0.5  # Penalty for slow resolution time

    # If system-level analysis is correct, reward, else penalize
    reward = 1.0 if system_level_correct else -1.0
    return reward + penalty + TTR_penalty

def mitigation_eval(result):
    # Check if mitigation was successful
    success = result["success"]

    # Check Time to Mitigation (TTM) for speed
    TTM_threshold = 20  # Define threshold for fast mitigation
    TTM_penalty = 0.0
    if result["TTM"] > TTM_threshold:
        TTM_penalty = -0.5  # Penalty for slow mitigation

    # Reward for success, otherwise penalize
    reward = 1.0 if success else -1.0
    return reward + TTM_penalty

def result_reward_func(results, **kwargs):
    rewards = []

    for result in results:
        if result["task"] == "detection":
            # Calculate reward for detection task
            reward = detection_eval(result)
            rewards.append(reward)
        elif result["task"] == "localization":
            # Calculate reward for localization task
            reward = localization_eval(result)
            rewards.append(reward)
        elif result["task"] == "analysis":
            # Calculate reward for root cause analysis task
            reward = analysis_eval(result)
            rewards.append(reward)
        elif result["task"] == "mitigation":
            # Calculate reward for mitigation task
            reward = mitigation_eval(result)
            rewards.append(reward)

    return rewards

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "format": format_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs

if __name__ == "__main__":
    # Example usage
    # rewards = format_reward(completions)
    # print(rewards)  # Output: [0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    # Example usage of format_reward
    rewards = format_reward(conversations, is_reasoning=True)
    print(rewards)

    # # Example usage of result_reward_func
    # result_rewards = result_reward_func(results)
    # print(result_rewards)  # Output: [1.0, -1.0, -1.0, -1.0]