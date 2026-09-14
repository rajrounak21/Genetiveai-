import os
from langchain.llms import HuggingFaceHub
from key import HUGGINGFACE_API_KEY
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Set environment variable for Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

# Streamlit Framework
st.title("Celebrity Search Result")
input_text = st.text_input("Search the topic you want")

# Define prompt templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about Celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world"
)

# Define memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='person_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='dob_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Define Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    model_kwargs={
        'max_new_tokens': 256,
        'temperature': 0.01
    },
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

# Define chains
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='person',
    memory=person_memory
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='dob',
    memory=dob_memory
)

chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='description',
    memory=descr_memory
)

# Define SequentialChain with correct input and output keys
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],  # Initial input
    output_variables=['description'],  # Final output
    verbose=True
)

# Run the chain if there is input
if input_text:
    result = parent_chain({'name': input_text})

    st.write(result)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
