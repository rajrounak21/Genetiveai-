   # Integrate Our code HUGGINGFACE API KEY

import  os
from langchain.llms import HuggingFaceHub
from key  import HUGGINGFACE_API_KEY
import streamlit as st
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACE_API_KEY

   # Streamlit Framework
st.title("Langchain Demo With HUGGINGFACE API ")
input_text = st.text_input("Search the topic you want")

  ## OPENAI LLMS
llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            model_kwargs={
                'max_new_tokens': 8000,#re this parameter matches your library ver# sion
               'temperature': 0.01
            },
            huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')  # Pass the API token directly
        )
submit=st.button(" Ask ")
if submit:
    st.write(llm(input_text))