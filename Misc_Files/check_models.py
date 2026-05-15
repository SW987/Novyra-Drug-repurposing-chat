import google.generativeai as genai
import os

# Configure with environment variable (secure)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY environment variable not set")
    print("   Please set it in your .env file or environment")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:")
for model in genai.list_models():
    print(f"- {model.name}: {model.description}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported methods: {model.supported_generation_methods}")
