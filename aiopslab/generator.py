import logging
from typing import Optional, List, Dict, Any

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
        aiopslab_host: str = "localhost",
        aiopslab_port: int = 8888,
        default_agent: str = "vllm",
        default_steps: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = AIOpsLabClient(f"http://{aiopslab_host}:{aiopslab_port}")
        self.default_agent = default_agent
        self.default_steps = default_steps

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (`List[Dict[str, Any]]`):
                Each item **must** contain at least a `"problem_id"` key. 
                Optional keys: `"agent_name"`, `"max_steps"`.
            n, repetition_penalty, temperature, top_p, top_k, min_p, max_tokens,
            guided_decoding_regex :  Same semantics as the vLLM generator.

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
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    guided_decoding_regex=guided_decoding_regex,
                )

                if response is None:
                    logger.warning("Simulation failed for problem_id=%s", problem_id)
                    continue
                conversations.append(
                    {
                        "agent": response["agent"],
                        "problem_id": response["problem_id"],
                        "trace": response["trace"],
                        "results": response["results"],
                    }
                )
                logger.info(
                    "Simulation completed for problem_id=%s, agent=%s",
                    problem_id,
                    response["agent"],
                )

        return conversations