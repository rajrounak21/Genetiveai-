import os
import google.generativeai as genai
from key import GOOGLE_API_KEY
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Setup your API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Select model
model = genai.GenerativeModel("gemini-3.6-flash")

# Streamlit app
st.title("LANGUAGE TRANSLATOR")

# User input
input_text = st.text_input("Enter your word or sentence", placeholder="e.g., what is your name?")
col1, col2 = st.columns([5, 5])

with col1:
    # Expanded language choices
    language_choice = st.selectbox(
        "Select Your Language",
        (
            "English", "Spanish", "Chinese", "Hindi", "Arabic", "Portuguese", "Bengali",
            "Russian", "Japanese", "Punjabi", "German", "Javanese", "Korean", "French",
            "Telugu", "Vietnamese", "Marathi", "Tamil", "Urdu", "Turkish", "Italian",
            "Yue Chinese", "Thai", "Gujarati", "Jin Chinese", "Southern Min", "Persian",
            "Polish", "Pashto", "Kannada", "Xiang Chinese", "Malayalam", "Sundanese",
            "Hausa", "Odia", "Burmese", "Hakka Chinese", "Ukrainian", "Bhojpuri", "Tagalog",
            "Yoruba", "Maithili", "Uzbek", "Sindhi", "Amharic", "Fula", "Romanian", "Oromo",
            "Igbo", "Azerbaijani", "Awadhi", "Dutch", "Kurdish", "Serbo-Croatian", "Malagasy",
            "Saraiki", "Nepali", "Sinhalese", "Chittagonian", "Zhuang", "Khmer", "Turkmen",
            "Assamese", "Madurese", "Somali", "Marwari", "Magahi", "Haryanvi", "Hungarian",
            "Chhattisgarhi", "Greek", "Chewa", "Deccan", "Akan", "Kazakh", "Min Bei Chinese",
            "Sylheti", "Zulu", "Czech", "Kinyarwanda", "Dhundhari", "Haitian Creole", "Ilocano",
            "Quechua", "Kirundi", "Swedish", "Hmong", "Shona", "Uyghur", "Hiligaynon", "Mossi",
            "Xhosa", "Belarusian", "Balochi", "Konkani", "Tswana", "Latvian", "Slovak", "Tigrinya"
        ),
        index=0
    )

# Define prompt templates
translation_prompt_template = PromptTemplate(
    input_variables=['Sentence', 'Language'],
    template="You are a language translator. Please translate only the given sentence to {Language}: {Sentence}."
)

# Function to call the Gemini API
def call_gemini_api(prompt):
    response = model.generate_content(prompt)
    if hasattr(response, 'text'):
        return response.text
    else:
        return 'No content returned from API.'

def create_chains(prompt_template):
    def chain_function(inputs):
        prompt = prompt_template.format(Sentence=inputs['Sentence'], Language=inputs['Language'])
        return call_gemini_api(prompt)
    return chain_function

# Initialize chain
chain = create_chains(translation_prompt_template)

# Action on translate button
if st.button("Translate"):
    if input_text:
        try:
            # Execute the chain with sentence and language choice
            result = chain({'Sentence': input_text, 'Language': language_choice})
            st.subheader("Your translated word or sentence:")
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please type a word or sentence to translate.")
