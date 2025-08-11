import os
import requests
import wikipedia
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from urllib.parse import quote_plus

# =============================
# Load API keys from .env file
# =============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Google Maps API key

# =============================
# Tools Section
# =============================

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city from the OpenWeather API.
    Returns a string like: 'Paris: sunny, 25°C'
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={quote_plus(city)}&appid={OPENWEATHER_API_KEY}&units=metric"
    print("Weather API URL:", url)
    response = requests.get(url)
    if response.status_code != 200:
        return f"Weather data not available for {city}."
    data = response.json()
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"{city}: {desc}, {temp}°C"


@tool
def get_coordinates(address: str) -> str:
    """
    Get latitude and longitude for an address using Google Maps Geocoding API.
    Returns a string 'lat,lon' or 'Coordinates not found.'
    """
    encoded_address = quote_plus(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={GOOGLE_MAPS_API_KEY}"
    print("Geocode API URL:", url)
    r = requests.get(url)
    if r.status_code != 200:
        return "Coordinates not found."
    result = r.json()
    if not result.get("results"):
        return "Coordinates not found."
    location = result["results"][0]["geometry"]["location"]
    return f"{location['lat']},{location['lng']}"


@tool
def get_drive_time_minutes(addresses: str) -> str:
    """
    Get driving time in minutes between two addresses using Google Maps Distance Matrix API.
    Expects input: "start_address|dest_address"
    Returns a string with the drive time in minutes (e.g., '45.2') or error message.
    """
    try:
        start_address, dest_address = [a.strip() for a in addresses.split("|")]
    except Exception:
        return "Invalid input format, expected 'start_address|dest_address'."

    start_encoded = quote_plus(start_address)
    dest_encoded = quote_plus(dest_address)

    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json?"
        f"origins={start_encoded}&destinations={dest_encoded}&mode=driving&key={GOOGLE_MAPS_API_KEY}"
    )
    print("Distance Matrix API URL:", url)

    r = requests.get(url)
    if r.status_code != 200:
        return "Error fetching distance data."
    data = r.json()

    try:
        duration_seconds = data["rows"][0]["elements"][0]["duration"]["value"]
    except (IndexError, KeyError):
        return "Driving time data not found."

    minutes = duration_seconds / 60
    return f"{minutes:.1f}"


@tool
def wiki_summary(place: str) -> str:
    """
    Get a 3-line Wikipedia summary for a place.
    """
    try:
        return wikipedia.summary(place, sentences=3)
    except Exception:
        return f"No Wikipedia info found for {place}."


# =============================
# LangChain Agent Setup
# =============================

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Register all tools
tools = [get_weather, get_coordinates, get_drive_time_minutes, wiki_summary]

# Use default ReAct prompt from LangChain hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct-style agent with tools
agent = create_react_agent(llm, tools, prompt)

# Create the executor that will run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# =============================
# Main Program
# =============================

if __name__ == "__main__":
    # Get user inputs
    home_address = input("Enter your home address: ").strip()
    city_name = input("Enter the city name: ").strip()

    # Instructions for the agent
    task = (
        f"Step 1: Call 'get_weather' for {city_name}.\n"
        f"Step 2: Based on the weather and your knowledge of {city_name}, suggest 5 attractions in the city that would be good to visit today.\n"
        f"Step 3: For each attraction, call 'get_drive_time_minutes' with input formatted as '{home_address}, {city_name}|<attraction>, {city_name}'.\n"
        "Step 4: Keep only those with a driving time of 60 minutes or less.\n"
        "Step 5: If fewer than 2 attractions qualify, suggest additional attractions and check again until you have at least 2 that meet the requirement.\n"
        "Step 6: Once you have 2 qualifying attractions, call 'wiki_summary' for each (3-line summary).\n"
        "Step 7: Return the weather, the two chosen attractions, their travel times, and the Wikipedia summaries."
    )

    # Run the agent
    result = agent_executor.invoke({"input": task})

    # Print final result
    print("\nFinal Result:\n", result["output"])
