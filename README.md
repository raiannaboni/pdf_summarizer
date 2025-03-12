# PDF Chat Assistant 📚

An interactive chat application that allows users to have conversations with their PDF documents. Built with Streamlit, LangChain, and OpenAI's GPT and DeepSeek models.
You can access it here: https://huggingface.co/spaces/raiannaboni/pdf_summarizer

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
- DeepSeek API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/raiannaboni/pdf_summarizer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy the `.env` file to create your own `.env` file:
```bash
cp .env.example .env
```

2. Add your API keys to the `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
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
- **LLM**: OpenAI's GPT and DeepSeek models for generating responses
- **RAG Implementation**: Uses LangChain's retrieval chain with history-aware retriever

## Contributing

Feel free to open issues and pull requests to improve the application.

## Credits

Created with ❤️ by Raianna Boni

Built with:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [DeepSeek](https://platform.deepseek.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
