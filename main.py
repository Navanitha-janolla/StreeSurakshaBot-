import os
import streamlit as st
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import TextLoader

knowledge_file = "C:\\Users\\Peddi\\OneDrive\\Desktop\\workchol ai\\knowledge1_base.txt"




# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\Peddi\\Downloads\\gen-lang-client-0304283798-b1c042008559.json"

def create_chatbot(knowledge_file):
    
    loader = TextLoader(knowledge_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"LLM Initialization Error: {e}")
        return None

    # Custom prompt to restrict answers to context only
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not in the context, say "Iam still learning about this".

    Context:
    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa

def run_streamlit_app(knowledge_file):
    st.title("🔍 StreeSurakshaBot 👧")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = create_chatbot(knowledge_file)

    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if query:
            if st.session_state.chatbot is not None:
                result = st.session_state.chatbot({"query": query})
                st.write("🤖 Answer:", result["result"])
            else:
                st.error("Chatbot initialization failed. See errors above.")
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    knowledge_file = "knowledge1_base.txt"
    run_streamlit_app(knowledge_file)

