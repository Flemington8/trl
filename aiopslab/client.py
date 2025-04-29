import json

import requests


class AIOpsLabClient:
    def __init__(self, base_url="http://localhost:8888"):
        self.base_url = base_url

    def list_problems(self):
        """Get list of available problems"""
        response = requests.get(f"{self.base_url}/problems")
        return response.json()

    def list_agents(self):
        """Get list of available agents"""
        response = requests.get(f"{self.base_url}/agents")
        return response.json()

    def run_simulation(
        self,
        problem_id,
        agent_name="vllm",
        max_steps=10,
        # vLLM specific parameters
        model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ):
        """
        Run a simulation with specified parameters.

        Args:
            problem_id: The ID of the problem to solve
            agent_name: Name of the agent to use (default: "vllm")
            max_steps: Maximum steps for the simulation

            # vLLM specific generation parameters (only used when agent_name is "vllm")
            n: Number of completions to generate
            repetition_penalty: Penalty for repeating tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            max_tokens: Maximum number of tokens to generate

        Returns:
            JSON response from the server or None if error.
            Example: {
                "agent": "Qwen2.5-Coder-3B-Instruct",
                "session_id": "1234567890",
                "problem_id": "misconfig_app_hotel_res-mitigation-1",
                "results": {
                    "Localization Accuracy": 0.0,
                    "success": false,
                    "is_subset": false,
                    "TTL": 0.209824800491333,
                    "steps": 10,
                    "in_tokens": 90,
                    "out_tokens": 160
                },
                "trace": [
                    {"role": "assistant", "content": "Action: exec_shell(\"kubectl get pods -n test-hotel-reservation\")"},
                    {"role": "env", "content": "Error parsing response: No API call found!"},
                    ...
                ]
            }
        """
        # Basic payload for any agent
        payload = {
            "problem_id": problem_id,
            "agent_name": agent_name,
            "max_steps": max_steps
        }

        # Add vLLM-specific parameters if the agent is vllm
        if agent_name == "vllm":
            vllm_params = {
                "model": model,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            # Add vLLM parameters to payload
            payload["vllm_params"] = vllm_params

        response = requests.post(
            f"{self.base_url}/simulate",
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None


# Example usage
if __name__ == "__main__":
    # Replace with your actual server IP or hostname
    client = AIOpsLabClient("http://localhost:8888")

    # List problems
    problems = client.list_problems()
    print(f"Available problems: {problems}")

    # List agents
    agents = client.list_agents()
    print(f"Available agents: {agents}")

    # Run a simulation
    result = client.run_simulation(
        problem_id="misconfig_app_hotel_res-mitigation-1",
        agent_name="deepseek",
        max_steps=10
    )

    if result:
        print(f"Simulation results:")
        print(f"Agent: {result['agent']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Problem: {result['problem_id']}")
        print(f"Results: {json.dumps(result['results'], indent=2)}")

        # Print first and last trace items
        if result['trace']:
            print("\nFirst trace item:")
            print(json.dumps(result['trace'][0], indent=2))

            if len(result['trace']) > 1:
                print("\nLast trace item:")
                print(json.dumps(result['trace'][-1], indent=2))
