from dotenv import load_dotenv
import os

load_dotenv()

NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")