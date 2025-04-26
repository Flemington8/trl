import requests
import json

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

    def run_simulation(self, problem_id, agent_name="deepseek", max_steps=10):
        """Run a simulation with specified parameters"""
        payload = {
            "problem_id": problem_id,
            "agent_name": agent_name,
            "max_steps": max_steps
        }
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
    client = AIOpsLabClient("http://137.184.6.93:8888")
    
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