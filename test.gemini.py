import google.generativeai as genai
genai.configure(api_key="AIzaSyBPumjvfIx5J8KkSTm_K0QWz_07jxbKa-Y")
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
response = model.generate_content("Hello, Gemini!")
print(response.text)