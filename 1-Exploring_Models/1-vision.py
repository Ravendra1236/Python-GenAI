from dotenv import load_dotenv 
load_dotenv()  # Loading all env variables 

import streamlit as st 
import os 
import google.generativeai as genai 
from PIL import Image 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


## Function to load  Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_reponse(input , image):
    if input != "":
        response = model.generate_content([input , image])
    else:
        response = model.generate_content(image) 
    return response.text  

## Setting up our streamlit app 

st.set_page_config(page_title="Q&A Demo") 
st.header("Gemini LLM Applicaton")

input = st.text_input("Input Prompt: ",key="input")

uploaded_file = st.file_uploader("Choose an image..." , type=["jpg" , "jpeg" , "png"])
image = "" 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image , caption="Upload Image." , use_container_width=True)
    
submit = st.button("Tell me about the image.")

#
if submit: 
    response = get_gemini_reponse(input , image) 
    st.subheader("The Response is")
    st.write(response)
