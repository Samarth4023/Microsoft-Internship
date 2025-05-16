from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
import os
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# API Key for weather
API_KEY = os.getenv("Weather_Token")

# -------------------- TOOL 1: Get Weather --------------------
@tool
def get_current_weather(place: str) -> str:
    """
    A tool that fetches the current weather of a particular place.

    Args:
        place (str): A string representing a valid place (e.g., 'London/Paris').

    Returns:
        str: Weather description including condition, temperature, humidity, and wind speed.
    """
    api_key = API_KEY
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": place,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather_desc = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            return (
                f"Weather in {place}:\n"
                f"- Condition: {weather_desc}\n"
                f"- Temperature: {temperature}°C\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind Speed: {wind_speed} m/s"
            )
        else:
            return f"Error: {data['message']}"
    except Exception as e:
        return f"Error fetching weather data for '{place}': {str(e)}"


# -------------------- TOOL 2: Get Time --------------------
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    A tool that fetches the current local time in a specified timezone.

    Args:
        timezone (str): A string representing a valid timezone (e.g., 'America/New_York').

    Returns:
        str: The current local time formatted as a string.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


# -------------------- TOOL 3: Document QnA --------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

@tool
def document_qna_tool(pdf_path: str, question: str) -> str:
    """
    A tool for answering questions based on the content of a PDF document.

    Args:
        pdf_path (str): Path to the local PDF file.
        question (str): A natural language question to ask about the PDF content.

    Returns:
        str: Answer to the question based on the PDF's content.
    """
    try:
        # Step 1: Extract text from PDF
        doc = fitz.open(pdf_path)
        text_chunks = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_chunks.append(text)
        doc.close()

        if not text_chunks:
            return "No text found in the PDF."

        # Step 2: Semantic search
        embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
        best_match_idx = scores.argmax()
        best_context = text_chunks[best_match_idx]

        # Step 3: Answer question
        prompt = f"Context: {best_context}\nQuestion: {question}"
        answer = qa_pipeline(prompt, max_new_tokens=100)[0]['generated_text']
        return f"Answer: {answer.strip()}"

    except Exception as e:
        return f"Error processing document QnA: {str(e)}"


# -------------------- Other Components --------------------
final_answer = FinalAnswerTool()
search_tool = DuckDuckGoSearchTool()

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        get_current_time_in_timezone,
        get_current_weather,
        image_generation_tool,
        search_tool,
        document_qna_tool,  # ← New Tool Added
        final_answer
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()