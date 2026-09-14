import streamlit as st
import os
from huggingface_hub import InferenceClient
from key import HUGGINGFACE_API_KEY

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

def getLLMAresponse(input_text, no_words, blog_style):
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or HUGGINGFACE_API_KEY
    if not api_key or not api_key.strip():
        return "Configuration error: HUGGINGFACE_API_KEY missing in key.py"

    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    client = InferenceClient(provider="auto", api_key=api_key)

    try:
        no_words_int = int(no_words)
    except:
        no_words_int = 500

    prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words_int} words."
    max_tokens = min(3000, no_words_int * 2)

    response = client.chat_completion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Generate Blogs", page_icon="🤖", layout="centered")
st.header("Generate Blogs")

input_text = st.text_input("Enter the Blog Topic", key="blog_topic")
col1, col2 = st.columns(2)
with col1:
    no_words = st.text_input("No of Words", value="500", key="no_words")
with col2:
    blog_style = st.selectbox("Writing the blog for", ('Researchers', 'Common People'), key="blog_style")

submit = st.button("Generate", key="generate_btn")

if submit:
    if not input_text.strip():
        st.error("Please enter a valid blog topic.")
    elif not no_words.isdigit():
        st.error("Please enter a valid number for No of Words.")
    else:
        with st.spinner("Generating blog... please wait ⏳"):
            try:
                result = getLLMAresponse(input_text, int(no_words), blog_style)
                st.success("Blog generated!")
                st.write(result)
            except Exception as e:
                st.error(f"Error generating response: {e}")
