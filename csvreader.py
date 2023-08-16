import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from apikey import apikey
from htmltemplate import css,user_template,bot_template


os.environ["OPENAI_API_KEY"] = apikey

# Define Streamlit app
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def app():
      # Title and description
    st.set_page_config(page_title="Chat With CSV", page_icon=":table:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    st.header("Chat With Multiple csv 📈")
        
    #st.write(css,unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents")
        file=st.file_uploader(
            "Upload Your Csv files here & click Process",type=["csv"]
        )
   
    if not file:
        st.stop()

    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head()) 

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 

    query = st.text_input("Enter a query:") 

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)


    
  
if __name__ == "__main__":
    app()   