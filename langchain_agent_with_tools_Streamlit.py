import os
import requests
import wikipedia
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# =============================
# Load API keys from .env file
# =============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# =============================
# Tools Section
# =============================

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city from OpenWeather API.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Weather data not available for {city}."
    data = response.json()
    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"{city}: {desc}, {temp}°C"


@tool
def get_drive_time_minutes(addresses: str) -> str:
    """
    Get driving time in minutes between two addresses using Google Maps Distance Matrix API.
    Expects input: "start_address|dest_address"
    Returns driving time in minutes as string or error message.
    """
    try:
        start_address, dest_address = [a.strip() for a in addresses.split("|")]
    except Exception:
        return "Invalid input format, expected 'start_address|dest_address'."

    # Prepare URL
    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={requests.utils.quote(start_address)}"
        f"&destinations={requests.utils.quote(dest_address)}"
        f"&mode=driving"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    print("Google Maps Distance Matrix URL:", url)  # debug print

    r = requests.get(url)
    if r.status_code != 200:
        return "Error getting driving time from Google Maps API."

    data = r.json()
    if data["status"] != "OK":
        return f"Error in API response: {data['status']}"

    try:
        element = data["rows"][0]["elements"][0]
        if element["status"] != "OK":
            return f"Route not found: {element['status']}"
        seconds = element["duration"]["value"]
        minutes = seconds / 60
        return f"{minutes:.1f}"
    except Exception:
        return "Unexpected API response format."


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

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_weather, get_drive_time_minutes, wiki_summary]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# =============================
# Streamlit UI
# =============================

st.title("Stockholm Attractions Planner 🌤️🏛️")

home_address = st.text_input("Enter your home address:")
city_name = st.text_input("Enter the city name:")

if st.button("Find Attractions"):
    if not home_address or not city_name:
        st.error("Please enter both your home address and city name.")
    else:
        task = (
            f"Step 1: Call 'get_weather' for {city_name}.\n"
            f"Step 2: Based on the weather and your knowledge of {city_name}, suggest 5 attractions in the city that would be good to visit today.\n"
            f"Step 3: For each attraction, call 'get_drive_time_minutes' with input formatted as '{home_address}, {city_name}|<attraction>, {city_name}'.\n"
            "Step 4: Keep only those with a driving time of 60 minutes or less.\n"
            "Step 5: If fewer than 2 attractions qualify, suggest additional attractions and check again until you have at least 2 that meet the requirement.\n"
            "Step 6: Once you have 2 qualifying attractions, call 'wiki_summary' for each (3-line summary).\n"
            "Step 7: Return the weather, the two chosen attractions, their travel times, and the Wikipedia summaries."
        )

        with st.spinner("Finding attractions..."):
            result = agent_executor.invoke({"input": task})
            st.subheader("Results:")
            st.write(result["output"])
