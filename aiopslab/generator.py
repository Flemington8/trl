import logging
from typing import Any, Dict, List
import json

from trl.extras.conversation_generator import ConversationGenerator
from .client import AIOpsLabClient

logger = logging.getLogger(__name__)


class AIOpsLabConversationGenerator(ConversationGenerator):
    """
    A ConversationGenerator that proxies every request to an AIOpsLab service
    instead of talking directly to a vLLM server.
    """

    def __init__(
        self,
        aiopslab_server_host: str = "0.0.0.0",
        aiopslab_server_port: int = 8888,
        model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        default_agent: str = "vllm",
        default_steps: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = AIOpsLabClient(
            f"http://{aiopslab_server_host}:{aiopslab_server_port}")
        self.model = model
        self.default_agent = default_agent
        self.default_steps = default_steps

    def _convert_trace_roles(self, trace):
        """
        Convert 'env' roles to 'user' roles for GRPO compatibility.
        """
        converted_trace = []
        
        for message in trace:
            # Convert env role to user role
            if message["role"] == "env":
                converted_trace.append({"role": "user", "content": message["content"]})
            else:
                converted_trace.append({"role": message["role"], "content": message["content"]})
                
        return converted_trace

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (`List[Dict[str, Any]]`):
                Each item **must** contain at least a `"problem_id"` key. 
                Optional keys: `"agent_name"`, `"max_steps"`.
            n (`int`, *optional*, defaults to `1`):
                Number of conversations to generate for each input.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.

        Returns (`List[Dict[str, Any]]`):
            One flattened conversation per *requested generation*,
            in the form expected by `GRPOTrainer()._prepare_conversations`.
        """
        conversations: List[Dict[str, Any]] = []

        for spec in inputs:
            problem_id = spec["problem_id"]
            agent_name = spec.get("agent_name", self.default_agent)
            max_steps = spec.get("max_steps", self.default_steps)
            for _ in range(n):
                response = self.client.run_simulation(
                    problem_id=problem_id,
                    agent_name=agent_name,
                    max_steps=max_steps,
                    model=self.model,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                )

                if response is None:
                    logger.warning(
                        "Simulation failed for problem_id=%s", problem_id)
                    continue
                conversation = {
                    "agent": response["agent"],
                    "task": spec["task"],
                    "problem_id": response["problem_id"],
                    "messages": self._convert_trace_roles(response["trace"]),
                    "result": response["results"],
                    }
                conversations.append(conversation)
                with open("./aiopslab/data/conversation.json", "w", encoding="utf-8") as f:
                    json.dump(conversation, f, ensure_ascii=False, indent=4)
                logger.info(
                    "Simulation completed for problem_id=%s, agent=%s",
                    problem_id,
                    response["agent"],
                )

        return conversations
