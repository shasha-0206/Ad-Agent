import os
import json
import google.generativeai as genai
from Processors import build_context
from Selector import select_agent
from InfoMiner import info_miner
from dotenv import load_dotenv

load_dotenv()

def setup_gemini(api_key: str, model: str = "gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    return model

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    model = setup_gemini(API_KEY)

    # Step 1: Run Processor
    user_cmd = input("Enter your command: ")
    processor_output = build_context(user_cmd, model)
    print("\nProcessor Output:\n", json.dumps(processor_output, indent=2))

    # Step 2: Run Selector using processor output
    selection = select_agent(processor_output, model=model)
    print("\nSelector Output:\n", json.dumps(selection, indent=2))

    # Step 3: Info Miner
    documentation = info_miner(selection, model_name=model)
    print("\nInfo Miner Output:\n", json.dumps(documentation, indent=2))