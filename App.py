from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
from PyPDF2 import PdfReader
dotenv.load_dotenv()

api_key = os.getenv('key')


# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85, google_api_key=api_key)

# Read and process PDF
pdfreader = PdfReader('./Corpus.pdf')
text = ''.join(page.extract_text() for page in pdfreader.pages)
docs = [Document(page_content=t) for t in text.split('\n\n')]

# Initialize embeddings and vectorstore
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=gemini_embeddings,
    persist_directory="./chroma_db"
)

# Load vectorstore from disk
vectorstore_disk = Chroma(
    persist_directory="./chroma_db",
    embedding_function=gemini_embeddings
)

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})
session_history = []

def update_session(question):
    global session_history
    session_history.append(question)

def get_session_context():
    global session_history
    return "\n".join(session_history)

llm_prompt_template = """You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.

Session Context:
{session_context}

Question: {question} 
Context: {context} 
Answer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# Format documents to string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Format input for the chain
def format_input(question):
    context = retriever | format_docs
    session_context = get_session_context()
    return {
        "context": context,
        "question": question,
        "session_context": session_context
    }

rag_chain = llm_prompt | llm | StrOutputParser()

st.title("Conversational Q&A Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_question = st.text_input("Type your message here...")
if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Update session history and get the answer
    update_session(user_question)
    formatted_input = format_input(user_question)
    answer = rag_chain.invoke(formatted_input)
    
    # Add assistant's answer to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Display assistant's message
    with st.chat_message("assistant"):
        st.markdown(answer)
