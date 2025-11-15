import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Configure the API key (it will be automatically picked up from environment variables if set)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-2.5-flash')

# Generate content
prompt = "Explain the concept of quantum entanglement in simple terms."
response = model.generate_content(prompt)

# Print the generated text
print(response.text)