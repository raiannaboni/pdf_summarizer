import streamlit as st
import faiss
import tempfile
import os
import time
import torch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# if not OPENAI_API_KEY:
#     st.error('Please set the OPENAI_API_KEY in your Streamlit secrets or .env file')
#     st.stop()

# Create a temporary directory for vector store
VECTOR_STORE_DIR = tempfile.mkdtemp()

# Starting Streamlit
st.set_page_config(
    page_title='Chat with PDFs 📚',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title('📚 Document Settings')
    st.markdown('---')
    
    uploads = st.file_uploader(
        label='Upload your PDFs',
        type=['pdf'],
        accept_multiple_files=True,
        help='You can upload multiple PDF files'
    )
    
    st.markdown('---')
    st.markdown('### Model Settings')
    model_class = st.selectbox(
        'Select Model:',
        ['openai'],
        index=0
    )
    
    
# Main content
st.title('💬 Chat with Your Documents')
st.markdown('Upload your PDFs and start asking questions about them!')

# LLM model
model_class = 'openai'
MODEL = 'gpt-4o-mini'


def model_openai(model='gpt-4o-mini', temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

    return llm


# Indexing and Retriever
def config_index_retriever(uploads):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

    # Store vectors in temporary directory
    vectorstore = FAISS.from_documents(splits, embeddings)
    vector_store_path = os.path.join(VECTOR_STORE_DIR, 'index')
    vectorstore.save_local(vector_store_path)

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 4}
    )

    return retriever


# Chain
def config_rag_chain(retriever):
    llm = model_openai()

    # Context Prompt
    context_q_system_prompt = 'Given the following chat history and the follow-up question which might reference context in the chat history, \
            formulate a standalone question which can be understood without the chat history. \
            Do NOT answer the question, just reformulate it if needed and otherwise return it as is.'
    
    context_q_user_prompt = 'Question: {input}'

    context_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ('system', context_q_system_prompt),
                            MessagesPlaceholder('chat_history'),
                            ('human', context_q_user_prompt)
                        ]
    )

    # Context Chain
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=context_q_prompt
    )

    # Q&A prompt
    qa_prompt_template = '''
            You are a helpful virtual assistant and are answering general questions.
            Use the following retrieved context pieces to answer the question.
            If you don''t know the answer, just say you don''t know. Keep the answer concise.
            Respond in the language requested by the user. If no language is specified, respond in English. \n\n
            Question: {input} \n
            Context: {context}
            '''
    
    qa_prompt = PromptTemplate.from_template(qa_prompt_template)

    # RAG Chain
    qa_chain = create_stuff_documents_chain(llm,
                                            qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,
                                       qa_chain)
    
    return rag_chain


# Streamlit app
if not uploads:
    st.info('Please upload a file')
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content='Hello, I''m your virtual assistant! How can I help you?')
    ]

if 'docs_list' not in st.session_state:
    st.session_state.docs_list = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('AI'):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.write(message.content)

start = time.time()
user_query = st.chat_input('Type your message here.')

if user_query is not None and user_query != '' and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('Human'):
        st.markdown(user_query)
    
    with st.chat_message('AI'):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            with st.spinner('Processing your documents...'):
                st.session_state.retriever = config_index_retriever(uploads)

        rag_chain = config_rag_chain(st.session_state.retriever)
        
        # Initialize the empty message placeholder
        message_placeholder = st.empty()
        full_response = ""

        # Stream the response
        for chunk in rag_chain.stream({
            'input': user_query,
            'chat_history': st.session_state.chat_history
        }):
            if 'answer' in chunk:
                full_response += chunk['answer']
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
       

    st.session_state.chat_history.append(AIMessage(content=full_response))

# Add footer
st.markdown('---')
st.markdown(
    '<div style="text-align: center">Made with ❤️ by Raianna Boni</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align: center">Using LangChain and Streamlit</div>',
    unsafe_allow_html=True
)









