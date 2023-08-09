from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain



def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask your PDF 🤔")
    
    pdf=st.file_uploader("Upload Your PDF", type="PDF")
    
    #upload pdf
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
        #slit pdfdata into multiple chunks
        text_splitter=CharacterTextSplitter(
          separator="\n",
          chunk_size=100,
          chunk_overlap=50,
          length_function=len
        )
        chuks=text_splitter.split_text(text)
        
        #create embedding
        embeddings=HuggingFaceEmbeddings()
        knowledge_base=FAISS.from_texts(chuks,embeddings)
        
        user_question=st.text_input("Ask your question PDF:")
        
        if user_question:
          docs = knowledge_base.similarity_search(user_question)
          llm=HuggingFaceHub()
          chain=load_qa_chain(llm,chain_type="stuff")
          with get_huggingface_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
              
          st.write(response)
          
               
          
    
            
    
if __name__=='__main__':
       main()
        