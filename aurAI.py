import asyncio
import os
from air import AsyncAIRefinery, DistillerClient, login
from dotenv import load_dotenv

load_dotenv()
auth = login(account=os.getenv("ACCOUNT"), api_key=os.getenv("API_KEY"))

# Agent 1: Analytics Agent
async def analytics_agent(query: str):
    prompt = f"Analyze historical data for demand spikes and supply delays based on: {query}"
    client = AsyncAIRefinery(**auth.openai())
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/Llama-3.1-70B-Instruct",
    )
    return response.choices[0].message.content

# Agent 2: Tool Use Agent
async def tool_use_agent(query: str):
    prompt = f"Connect to ERP and CRM systems to retrieve real-time data relevant to: {query}"
    client = AsyncAIRefinery(**auth.openai())
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/Llama-3.1-70B-Instruct",
    )
    return response.choices[0].message.content

# Agent 3: Planning Agent
async def planning_agent(query: str):
    prompt = f"Using insights from analytics and real-time data, generate a revised forecast, replenishment plan, and scenario strategy for: {query}"
    client = AsyncAIRefinery(**auth.openai())
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/Llama-3.1-70B-Instruct",
    )
    return response.choices[0].message.content

# Main runner
async def run_aurAI():
    distiller_client = DistillerClient()
    distiller_client.create_project(config_path="aurAI.yaml", project="aurAI")

    executor_dict = {
        "Analytics Agent": analytics_agent,
        "Tool Use Agent": tool_use_agent,
        "Planning Agent": planning_agent,
    }

    async with distiller_client(project="aurAI", uuid="planner_user", executor_dict=executor_dict) as dc:
        query = input("Describe your supply chain issue: ")  # 👈 dynamic input
        responses = await dc.query(query=query)
        async for response in responses:
            print(response['content'])

if __name__ == "__main__":
    asyncio.run(run_aurAI())
