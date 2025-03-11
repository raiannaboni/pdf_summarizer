# PDF Chat Assistant 📚

An interactive chat application that allows users to have conversations with their PDF documents. Built with Streamlit, LangChain, and OpenAI's GPT model.

## Features

- 📄 Multiple PDF document upload support
- 💬 Interactive chat interface
- 🤖 Context-aware responses using RAG (Retrieval Augmented Generation)
- 🌍 Multi-language support
- 🎨 Clean and intuitive user interface
- 💾 Efficient document processing with FAISS vector store
- 🔄 Streaming responses for better user experience

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Hugging Face API key (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
HF_API_KEY=your_huggingface_api_key
```

## Running the Application

To run the application locally:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. Upload one or more PDF files using the sidebar
2. Wait for the documents to be processed
3. Start asking questions about your documents in the chat interface
4. Receive AI-generated responses based on the content of your PDFs

## Technical Details

- **Document Processing**: Uses RecursiveCharacterTextSplitter for efficient text chunking
- **Embeddings**: Utilizes BAAI/bge-m3 model from Hugging Face for document embeddings
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: OpenAI's GPT model for generating responses
- **RAG Implementation**: Uses LangChain's retrieval chain with history-aware retriever

## Contributing

Feel free to open issues and pull requests to improve the application.

## Credits

Created with ❤️ by Raianna Boni

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)