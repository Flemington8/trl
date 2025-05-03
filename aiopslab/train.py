from datasets import Dataset
from peft import LoraConfig

from aiopslab.generator import AIOpsLabConversationGenerator
from aiopslab.rewards import format_reward, result_reward
from trl.trainer import GRPOConfig, GRPOTrainer

from dotenv import load_dotenv
from datetime import datetime

from transformers.integrations import WandbCallback

class PatchedWandb(WandbCallback):
    """W&B callback with the 'fake Trainer' on_train_end step disabled."""
    def on_train_end(self, *_, **__):
        # do nothing → no second Accelerator is constructed
        return

dataset = Dataset.from_list([
    {"problem_id": "k8s_target_port-misconfig-detection-1", "task": "detection"},
    {"problem_id": "k8s_target_port-misconfig-localization-1", "task": "localization"},
    {"problem_id": "k8s_target_port-misconfig-analysis-1", "task": "analysis"},
    {"problem_id": "k8s_target_port-misconfig-mitigation-1", "task": "mitigation"},
    {"problem_id": "k8s_target_port-misconfig-detection-2", "task": "detection"},
    {"problem_id": "k8s_target_port-misconfig-localization-2", "task": "localization"},
    {"problem_id": "k8s_target_port-misconfig-analysis-2", "task": "analysis"},
    {"problem_id": "k8s_target_port-misconfig-mitigation-2", "task": "mitigation"},
    {"problem_id": "k8s_target_port-misconfig-detection-3", "task": "detection"},
    {"problem_id": "k8s_target_port-misconfig-localization-3", "task": "localization"},
    {"problem_id": "k8s_target_port-misconfig-analysis-3", "task": "analysis"},
    {"problem_id": "k8s_target_port-misconfig-mitigation-3", "task": "mitigation"},

    {"problem_id": "auth_miss_mongodb-detection-1", "task": "detection"},
    {"problem_id": "auth_miss_mongodb-localization-1", "task": "localization"},
    {"problem_id": "auth_miss_mongodb-analysis-1", "task": "analysis"},
    {"problem_id": "auth_miss_mongodb-mitigation-1", "task": "mitigation"},

    {"problem_id": "revoke_auth_mongodb-detection-1", "task": "detection"},
    {"problem_id": "revoke_auth_mongodb-localization-1", "task": "localization"},
    {"problem_id": "revoke_auth_mongodb-analysis-1", "task": "analysis"},
    {"problem_id": "revoke_auth_mongodb-mitigation-1", "task": "mitigation"},
    {"problem_id": "revoke_auth_mongodb-detection-2", "task": "detection"},
    {"problem_id": "revoke_auth_mongodb-localization-2", "task": "localization"},
    {"problem_id": "revoke_auth_mongodb-analysis-2", "task": "analysis"},
    {"problem_id": "revoke_auth_mongodb-mitigation-2", "task": "mitigation"},

    {"problem_id": "user_unregistered_mongodb-detection-1", "task": "detection"},
    {"problem_id": "user_unregistered_mongodb-localization-1", "task": "localization"},
    {"problem_id": "user_unregistered_mongodb-analysis-1", "task": "analysis"},
    {"problem_id": "user_unregistered_mongodb-mitigation-1", "task": "mitigation"},
    {"problem_id": "user_unregistered_mongodb-detection-2", "task": "detection"},
    {"problem_id": "user_unregistered_mongodb-localization-2", "task": "localization"},
    {"problem_id": "user_unregistered_mongodb-analysis-2", "task": "analysis"},
    {"problem_id": "user_unregistered_mongodb-mitigation-2", "task": "mitigation"},

    {"problem_id": "misconfig_app_hotel_res-detection-1", "task": "detection"},
    {"problem_id": "misconfig_app_hotel_res-localization-1", "task": "localization"},
    {"problem_id": "misconfig_app_hotel_res-analysis-1", "task": "analysis"},
    {"problem_id": "misconfig_app_hotel_res-mitigation-1", "task": "mitigation"},

    {"problem_id": "scale_pod_zero_social_net-detection-1", "task": "detection"},
    {"problem_id": "scale_pod_zero_social_net-localization-1", "task": "localization"},
    {"problem_id": "scale_pod_zero_social_net-analysis-1", "task": "analysis"},
    {"problem_id": "scale_pod_zero_social_net-mitigation-1", "task": "mitigation"},
    
    {"problem_id": "assign_to_non_existent_node_social_net-detection-1", "task": "detection"},
    {"problem_id": "assign_to_non_existent_node_social_net-localization-1", "task": "localization"},
    {"problem_id": "assign_to_non_existent_node_social_net-analysis-1", "task": "analysis"},
    {"problem_id": "assign_to_non_existent_node_social_net-mitigation-1", "task": "mitigation"},
    
    {"problem_id": "container_kill-detection", "task": "detection"},
    {"problem_id": "container_kill-localization", "task": "localization"},

    {"problem_id": "pod_failure_hotel_res-detection-1", "task": "detection"},
    {"problem_id": "pod_failure_hotel_res-localization-1", "task": "localization"},

    {"problem_id": "pod_kill_hotel_res-detection-1", "task": "detection"},
    {"problem_id": "pod_kill_hotel_res-localization-1", "task": "localization"},

    {"problem_id": "network_loss_hotel_res-detection-1", "task": "detection"},
    {"problem_id": "network_loss_hotel_res-localization-1", "task": "localization"},

    {"problem_id": "network_delay_hotel_res-detection-1", "task": "detection"},
    {"problem_id": "network_delay_hotel_res-localization-1", "task": "localization"},

    {"problem_id": "noop_detection_hotel_reservation-1", "task": "detection"},
    {"problem_id": "noop_detection_social_network-1", "task": "detection"},
    {"problem_id": "noop_detection_astronomy_shop-1", "task": "detection"},

    {"problem_id": "astronomy_shop_ad_service_failure-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_ad_service_failure-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_ad_service_high_cpu-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_ad_service_high_cpu-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_ad_service_manual_gc-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_ad_service_manual_gc-localization-1", "task": "localization"},
    
    {"problem_id": "astronomy_shop_cart_service_failure-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_cart_service_failure-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_image_slow_load-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_image_slow_load-localization-1", "task": "localization"},
    
    {"problem_id": "astronomy_shop_kafka_queue_problems-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_kafka_queue_problems-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_loadgenerator_flood_homepage-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_loadgenerator_flood_homepage-localization-1", "task": "localization"},
    
    {"problem_id": "astronomy_shop_payment_service_failure-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_payment_service_failure-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_payment_service_unreachable-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_payment_service_unreachable-localization-1", "task": "localization"},
    
    {"problem_id": "astronomy_shop_product_catalog_service_failure-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_product_catalog_service_failure-localization-1", "task": "localization"},
    {"problem_id": "astronomy_shop_recommendation_service_cache_failure-detection-1", "task": "detection"},
    {"problem_id": "astronomy_shop_recommendation_service_cache_failure-localization-1", "task": "localization"},
    
    {"problem_id": "redeploy_without_PV-detection-1", "task": "detection"},
    {"problem_id": "redeploy_without_PV-analysis-1", "task": "analysis"},
    {"problem_id": "redeploy_without_PV-mitigation-1", "task": "mitigation"},

    {"problem_id": "wrong_bin_usage-detection-1", "task": "detection"},
    {"problem_id": "wrong_bin_usage-localization-1", "task": "localization"},
    {"problem_id": "wrong_bin_usage-analysis-1", "task": "analysis"},
    {"problem_id": "wrong_bin_usage-mitigation-1", "task": "mitigation"}
])

# Load environment variables from .env file
load_dotenv()

current_time = datetime.now().strftime("%m%d_%H%M")
run_name = f"Qwen2.5-Coder-0.5B-Instruct-GRPO-AIOpsLab-{current_time}"

training_args = GRPOConfig(num_train_epochs=1,
                           max_completion_length=1024,
                           temperature=1.0,
                           top_p=0.95,
                           # GRPO-specific parameters
                           num_generations=2,
                           num_iterations=2,
                           beta=0.2,
                           epsilon=0.2,
                           # Training parameters
                           fp16=True,
                           per_device_train_batch_size=1,
                           gradient_accumulation_steps=2,
                           is_conversation=True,
                           output_dir=f"./results/Qwen2.5-Coder-0.5B-Instruct-GRPO-AIOpsLab-{current_time}",
                           report_to=["wandb"],
                           run_name=run_name
                           )

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

conversation_generator = AIOpsLabConversationGenerator(
    vllm_server_host="0.0.0.0",
    vllm_server_port=8000,
    vllm_server_timeout=240.0,
    aiopslab_server_host="localhost",
    aiopslab_server_port=8888,
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    reward_funcs=[format_reward, result_reward],
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    conversation_generator=conversation_generator,
)

trainer.remove_callback(WandbCallback)   # remove the original
trainer.add_callback(PatchedWandb())     # add the safe version
trainer.train()
