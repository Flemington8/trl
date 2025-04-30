import re
from typing import Union, Any

MAX_STEPS = 10.0

def format_reward(conversations: list[dict[str, Union[str, Any]]], **kwargs):
    """
    Reward function that evaluates conversations based on the format correctness
    of completions.
    
    Args:
        conversations: List of conversation dictionaries, each must containing a "messages" list
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

    TTD_reward = 0.0
    steps_reward = 0.0

    if detection_correct:
        # Check the detection time (TTD) is within acceptable limits
        TTD_threshold = 50  # Define threshold for fast detection
        if result["TTD"] < TTD_threshold:
            TTD_reward = 0.5  # Reward for short detection time
        
        # Check the number of steps taken
        steps_reward = (MAX_STEPS - float(result["steps"])) / MAX_STEPS

    # If detection is correct, reward, else penalize
    reward = 1.0 if detection_correct else -1.0
    return (reward + TTD_reward + steps_reward) * 10.0

def localization_eval(result):
    # Check the localization accuracy
    localization_correct = result["success"]

    TTL_reward = 0.0
    steps_reward = 0.0

    if localization_correct:
        # Check the localization time (TTL)
        TTL_threshold = 40  # Define threshold for fast localization
        if result["TTL"] < TTL_threshold:
            TTL_reward = 0.5  # Reward for short localization time
        
        # Reward fewer steps (just like in detection_eval)
        steps_reward = (MAX_STEPS - float(result["steps"])) / MAX_STEPS

    accuracy_upbound = 100.0
    # Base reward based on correctness
    reward = (result["Localization Accuracy"] - accuracy_upbound) / accuracy_upbound
    return (reward + TTL_reward + steps_reward) * 10.0

def analysis_eval(result):
    # Check if system-level analysis is correct
    system_level_correct = result["system_level_correct"]
    fault_type_correct = result["fault_type_correct"]
    
    # Determine overall correctness
    analysis_correct = system_level_correct and fault_type_correct
    
    TTA_reward = 0.0
    steps_reward = 0.0
    
    if analysis_correct:
        # Check Time to Analysis (TTA)
        TTA_threshold = 40
        if result["TTA"] < TTA_threshold:
            TTA_reward = 0.5  # Reward for quick analysis
            
        # Reward fewer steps
        steps_reward = (MAX_STEPS - float(result["steps"])) / MAX_STEPS
    
    # Base reward based on system-level analysis correctness
    reward = 1.0 if system_level_correct else -1.0
    
    # Penalty for incorrect fault type (only if system analysis was correct)
    fault_type_penalty = -0.5 if (system_level_correct and not fault_type_correct) else 0.0
    
    return (reward + fault_type_penalty + TTA_reward + steps_reward) * 10.0

def mitigation_eval(result):
    # Check if mitigation was successful
    mitigation_correct = result["success"]
    
    TTM_reward = 0.0
    steps_reward = 0.0
    
    if mitigation_correct:
        # Check Time to Mitigation (TTM)
        TTM_threshold = 100
        if result["TTM"] < TTM_threshold:
            TTM_reward = 0.5  # Reward for fast mitigation
        
        # Reward fewer steps
        steps_reward = (MAX_STEPS - float(result["steps"])) / MAX_STEPS
    
    # Base reward based on success
    reward = 1.0 if mitigation_correct else -1.0
    return (reward + TTM_reward + steps_reward) * 10.0

def result_reward(conversations: list[dict[str, Union[str, Any]]], **kwargs):
    """
    Reward function that evaluates conversations based on the format correctness
    of completions.
    
    Args:
        conversations: List of conversation dictionaries, each must containing a "results" list
        **kwargs: Additional parameters, including is_reasoning flag
        
    Returns:
        List of reward scores, one per conversation
    """
    rewards = []

    for conversation in conversations:
        if conversation["task"] == "detection":
            # Calculate reward for detection task
            reward = detection_eval(conversation["result"])
            rewards.append(reward)
        elif conversation["task"] == "localization":
            # Calculate reward for localization task
            reward = localization_eval(conversation["result"])
            rewards.append(reward)
        elif conversation["task"] == "analysis":
            # Calculate reward for root cause analysis task
            reward = analysis_eval(conversation["result"])
            rewards.append(reward)
        elif conversation["task"] == "mitigation":
            # Calculate reward for mitigation task
            reward = mitigation_eval(conversation["result"])
            rewards.append(reward)

    return rewards

if __name__ == "__main__":
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "What is the problem?"},
                {"role": "assistant", "content": "<think>...</think><answer>```\nexec_shell(\"ls\")\n```"},
                {"role": "user", "content": "What is the solution?"},
                {"role": "assistant", "content": "<think>...</think><answer>```\nsubmit(\"solution\")\n```"}
            ],
            "task": "detection",
            "result": {
                "Detection Accuracy": "Correct",
                "TTD": 105.1446475982666,
                "steps": 4,
                "in_tokens": 10152,
                "out_tokens": 73
            }
        },
        {
            "messages": [
                {"role": "user", "content": "What is the problem?"},
                {"role": "assistant", "content": "<think>...</think><answer>```\nexec_shell(\"ls\")\n```"},
                {"role": "user", "content": "What is the solution?"},
                {"role": "assistant", "content": "<think>...</think><answer>```\nsubmit(\"solution\")\n```"}
            ],
            "task": "localization",
            "result": {
                "Localization Accuracy": 0.0,
                "TTL": 20.57143259048462,
                "steps": 5,
                "in_tokens": 2187,
                "out_tokens": 140,
                "success": False,
                "is_subset": False
            }
        }
    ]
    # Example usage of format_reward
    rewards = format_reward(conversations, is_reasoning=True)
    print(rewards)

    # Example usage of result_reward
    result_rewards = result_reward(conversations)
    print(result_rewards)