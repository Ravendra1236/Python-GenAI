# CWMP : Chat with multiple pdf



import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai 
import os

load_dotenv() 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
    return text

def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap=1000)
    chunks = text_splitter.split_text(text) 
    return chunks 



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore
    


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Add allow_dangerous_deserialization flag for trusted local files
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])
    

    
def main():
    st.header("Chat with PDF 💭")
    
    # Upload PDF
    pdf_docs = st.file_uploader("Upload your PDF Files", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Process"):
        with st.spinner("Processing..."):
            # Get PDF text
            raw_text = get_pdf_text(pdf_docs)
            
            # Get text chunks
            text_chunks = get_text_chunks(raw_text)
            
            # Create vector store
            vectorstore = get_vector_store(text_chunks)
            st.success("Done!")
            
    # Get user question
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
    
