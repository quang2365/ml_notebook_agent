from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(override=True)
llm = ChatOpenAI(model="nvidia/nemotron-3-ultra-550b-a55b", base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"))