import os
import requests
import wikipedia
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ===== Tools =====
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    print("url to weather" )
    print(url)
    response = requests.get(url)
    if response.status_code != 200:
        return f"Weather data not available for {city}."
    data = response.json()
    print("printing weather response data")
    print (data)
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"{city}: {desc}, {temp}°C"

@tool
def wiki_summary(place: str) -> str:
    """Get a 3-line Wikipedia summary for a place."""
    try:
        return wikipedia.summary(place, sentences=3)
    except Exception:
        return f"No Wikipedia info found for {place}."

# ===== LLM =====
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ===== Tools list =====
tools = [get_weather, wiki_summary]

# ===== Use LangChain's default ReAct prompt from the hub =====
prompt = hub.pull("hwchase17/react")  # contains {tools}, {tool_names}, {agent_scratchpad}

# ===== Create the agent and executor =====
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ===== Run =====
if __name__ == "__main__":
    city_name = input("Enter the city name: ").strip()
    task = (
        f"First, call the 'get_weather' tool to get the current weather for {city_name}. "
        "Based on that weather and your own knowledge of the city's attractions, pick the top two attractions "
        "to visit today. Then call the 'wiki_summary' tool for each to give a 3-line summary."
    )
    result = agent_executor.invoke({"input": task})
    print("\nResult:\n", result["output"])
