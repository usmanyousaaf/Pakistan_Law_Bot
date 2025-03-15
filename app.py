import streamlit as st
import os
import tempfile
import uuid
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import re

# Custom CSS Injection
def inject_custom_css():
    st.markdown("""
        <style>
            /* Main container */
            .stApp {
                background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
                color: #e0e0e0;
            }
            
            /* Chat containers */
            .stChatMessage {
                padding: 1.5rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* User message styling */
            [data-testid="stChatMessage"][aria-label="user"] {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                margin-left: 10%;
            }
            
            /* Assistant message styling */
            [data-testid="stChatMessage"][aria-label="assistant"] {
                background-color: #004d40;
                border: 1px solid #00695c;
                margin-right: 10%;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: #121212 !important;
                border-right: 2px solid #2d2d2d;
                padding: 1rem;
            }
            
            /* Button styling */
            .stButton>button {
                background: linear-gradient(45deg, #00695c, #004d40);
                color: white !important;
                border: none;
                border-radius: 8px;
                padding: 0.8rem 1.5rem;
                transition: all 0.3s;
                font-weight: 500;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                border: 2px dashed #3d3d3d;
                border-radius: 10px;
                padding: 1rem;
                background: #2d2d2d;
            }
            
            /* Input field */
            .stTextInput>div>div>input {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                padding: 0.8rem;
            }
            
            /* Spinner color */
            .stSpinner>div>div {
                border-color: #00bcd4 transparent transparent transparent;
            }
            
            /* Custom title styling */
            .title-text {
                background: linear-gradient(45deg, #00bcd4, #00695c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Roboto', sans-serif;
                font-size: 2.8rem;
                text-align: center;
                margin-bottom: 2rem;
                letter-spacing: -0.5px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            /* Similar questions buttons */
            .stButton>button.similar-q {
                background: #2d2d2d;
                border: 1px solid #00bcd4;
                color: #00bcd4 !important;
                white-space: normal;
                height: auto;
                min-height: 3rem;
                transition: all 0.3s;
            }
            
            /* Hover effects */
            .stButton>button.similar-q:hover {
                background: #004d40 !important;
                transform: scale(1.02);
            }
            
            /* Source text styling */
            .source-text {
                color: #00bcd4;
                font-size: 0.9rem;
                margin-top: 1rem;
                padding-top: 0.5rem;
                border-top: 1px solid #3d3d3d;
            }
        </style>
    """, unsafe_allow_html=True)

# Page Configuration
st.set_page_config(
    page_title="AI Law Agent",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"
DEFAULT_DOCUMENT_PATH = "lawbook.pdf"
DEFAULT_COLLECTION_NAME = "pakistan_laws_default"
CHROMA_PERSIST_DIR = "./chroma_db"

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "similar_questions" not in st.session_state:
    st.session_state.similar_questions = []
if "using_custom_docs" not in st.session_state:
    st.session_state.using_custom_docs = False
if "custom_collection_name" not in st.session_state:
    st.session_state.custom_collection_name = f"custom_laws_{st.session_state.user_id}"

def setup_embeddings():
    """Sets up embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def setup_llm():
    """Setup the language model"""
    if st.session_state.llm is None:
        st.session_state.llm = ChatGroq(
            model_name=MODEL_NAME, 
            groq_api_key=DEFAULT_GROQ_API_KEY,
            temperature=0.2
        )
    return st.session_state.llm

def check_default_db_exists():
    """Check if the default document database already exists"""
    return os.path.exists(os.path.join(CHROMA_PERSIST_DIR, DEFAULT_COLLECTION_NAME))

def load_existing_vectordb(collection_name):
    """Load an existing vector database from disk"""
    embeddings = setup_embeddings()
    try:
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    except Exception as e:
        st.error(f"Error loading existing database: {str(e)}")
        return None

def process_default_document(force_rebuild=False):
    """Process the default Pakistan laws document"""
    if check_default_db_exists() and not force_rebuild:
        st.info("Loading existing Pakistan law database...")
        db = load_existing_vectordb(DEFAULT_COLLECTION_NAME)
        if db:
            st.session_state.vectordb = db
            setup_qa_chain()
            st.session_state.using_custom_docs = False
            return True
    
    if not os.path.exists(DEFAULT_DOCUMENT_PATH):
        st.error(f"Default document {DEFAULT_DOCUMENT_PATH} not found.")
        return False
    
    try:
        with st.spinner("Building Pakistan law database..."):
            loader = PyPDFLoader(DEFAULT_DOCUMENT_PATH)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = "Pakistan Laws (Official)"
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            db = Chroma.from_documents(
                documents=chunks,
                embedding=setup_embeddings(),
                collection_name=DEFAULT_COLLECTION_NAME,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
            db.persist()
            st.session_state.vectordb = db
            setup_qa_chain()
            st.session_state.using_custom_docs = False
            return True
    except Exception as e:
        st.error(f"Error processing default document: {str(e)}")
        return False

def process_custom_documents(uploaded_files):
    """Process user-uploaded PDF documents"""
    embeddings = setup_embeddings()
    collection_name = st.session_state.custom_collection_name
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
            
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.name
                
            documents.extend(file_docs)
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        with st.spinner("Building custom document database..."):
            if Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
                collection_name=collection_name
            ).get():
                Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                    collection_name=collection_name
                ).delete_collection()
                
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=CHROMA_PERSIST_DIR
            )
            
            db.persist()
            st.session_state.vectordb = db
            setup_qa_chain()
            st.session_state.using_custom_docs = True
            return True
    return False

def setup_qa_chain():
    """Set up the QA chain with the RAG system"""
    if st.session_state.vectordb:
        template = """You are a helpful legal assistant specializing in Pakistani law. 
        Use the context to answer. If unsure, say so but provide general info.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=setup_llm(),
            chain_type="stuff",
            retriever=st.session_state.vectordb.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": ChatPromptTemplate.from_template(template)},
            return_source_documents=True
        )

def generate_similar_questions(question, docs):
    """Generate similar questions based on retrieved documents"""
    llm = setup_llm()
    context = "\n".join([doc.page_content for doc in docs[:2]])
    
    prompt = f"""Generate 3 similar questions based on:
    Original Question: {question}
    Legal Context: {context}
    Generate exactly 3 similar questions:"""
    
    try:
        response = llm.invoke(prompt)
        questions = re.findall(r"\d+\.\s+(.*?)(?=\d+\.|$)", response.content, re.DOTALL)
        return [q.strip() for q in questions[:3] if "?" in q]
    except Exception as e:
        return []

def get_answer(question):
    """Get answer from QA chain"""
    if not st.session_state.vectordb:
        with st.spinner("Loading Pakistan law database..."):
            process_default_document()
    
    if st.session_state.qa_chain:
        result = st.session_state.qa_chain({"query": question})
        answer = result["result"]
        sources = set()
        
        for doc in result.get("source_documents", []):
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
        
        if sources:
            answer += f"\n\nSources: {', '.join(sources)}"
            
        st.session_state.similar_questions = generate_similar_questions(
            question, result.get("source_documents", [])
        )
        return answer
    return "Initializing knowledge base..."

def main():
    inject_custom_css()  # CSS injection added here
    st.title("Pakistan Law AI Agent ⚖️")
    
    if st.session_state.using_custom_docs:
        st.subheader("Training on your personal resources")
    else:
        st.subheader("Powered by Pakistan law database")
    
    with st.sidebar:
        st.header("Resource Management")
        
        if st.session_state.using_custom_docs:
            if st.button("Return to Official Database"):
                with st.spinner("Loading official database..."):
                    process_default_document()
                    st.session_state.messages.append(AIMessage(content="Switched to official database!"))
                    st.rerun()
        
        if not st.session_state.using_custom_docs:
            if st.button("Rebuild Official Database"):
                with st.spinner("Rebuilding..."):
                    process_default_document(force_rebuild=True)
                    st.rerun()
        
        st.header("Upload Custom Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Train on Uploaded Documents") and uploaded_files:
            with st.spinner("Processing..."):
                if process_custom_documents(uploaded_files):
                    st.session_state.messages.append(AIMessage(content="Custom documents loaded!"))
                    st.rerun()

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.write(message.content)

    if st.session_state.similar_questions:
        st.markdown("#### Related Questions:")
        cols = st.columns(len(st.session_state.similar_questions))
        for i, q in enumerate(st.session_state.similar_questions):
            if cols[i].button(q, key=f"similar_q_{i}"):
                st.session_state.messages.extend([
                    HumanMessage(content=q),
                    AIMessage(content=get_answer(q))
                ])
                st.rerun()

    if user_input := st.chat_input("Ask a legal question..."):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Thinking..."):
                response = get_answer(user_input)
            st.write(response)
        
        st.session_state.messages.append(AIMessage(content=response))
        st.rerun()

if __name__ == "__main__":
    main()