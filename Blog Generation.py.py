import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub

# Import your API key securely
from key import HUGGINGFACE_API_KEY

# Ensure the environment variable is set
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

def getLLMAresponse(input_text, no_words, blog_style):
    try:
        # Initialize the LLM with appropriate parameters
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            model_kwargs={
                'max_new_tokens': 256,  # Ensure this parameter matches your library version
                'temperature': 0.01
            },
            huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')  # Pass the API token directly
        )
    except Exception as e:
        return f"Configuration error: {e}"

    # Define the prompt template
    template = """Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."""
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Format the prompt and generate the response
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    try:
        response = llm(formatted_prompt)
    except Exception as e:
        return f"Error generating response: {e}"

    # Return the response
    return response

# Streamlit app setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs")

# Input fields for blog generation
input_text = st.text_input("Enter the Blog Topic")

# Creating columns for additional inputs
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words")
    no_words = int(no_words) if no_words.isdigit() else 300  # Default to 300 if not a valid number

with col2:
    blog_style = st.selectbox(
        "Writing the blog for",
        ('Researchers', 'Data Scientist', 'Common People'),
        index=0
    )

submit = st.button("Generate")

if submit:
    response = getLLMAresponse(input_text, no_words, blog_style)
    st.write(response)
