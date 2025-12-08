import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyBDd4K-NL86geJG5Mz9r11YF4bOBRf6jb4")

print("Available Gemini models:")
for model in genai.list_models():
    print(f"- {model.name}: {model.description}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported methods: {model.supported_generation_methods}")
