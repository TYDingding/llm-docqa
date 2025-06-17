# PDF Q&A Chatbot with LangChain + OpenAI

This project is a chatbot that can answer questions about local PDF using **LangChain**, **OpenAI GPT**, and **FAISS**. It demonstrates a Retrieval-Augmented Generation (RAG) workflow, where your question is answered based on the contents of a document.

## Features

- Upload and query local PDF files
- Embeds documents using OpenAI Embeddings
- Retrieves relevant chunks using FAISS vector store
- Generates answers using ChatGPT API (gpt-4.1-nano)

## Setup

1. Clone the Repo

   ```
   clone git@github.com:your-username/llm-docqa.git
   cd llm-docqa
   ```

2. Install dependencies

   ```bash
   cd llm-docqa
   pip install -r requirements.txt
   ```

3. Add your OpenAI API Key

   Create a .env file:

   ```bash
   cp .env.example .env
   ```

   Edit .env:

   ```ini
   OPENAI_API_KEY=your-key-here
   ```

4. Add your PDF
   Place your PDF file inside the pdfs/ folder and rename it sample.pdf (or change the filename in main.py).

## Run The App

```bash
python main.py
```

Example:

```vbnet
You: What is this pdf about?

DocQA: This PDF appears to be a scholarly article from the proceedings of the 2008 International Snow Science Workshop. It discusses avalanche survival strategies, particularly focusing on different parts of a flowing avalanche and how individuals can improve their chances of survival. The paper includes a practitioner's perspective, theoretical insights into avalanche dynamics, case studies of avalanche survivors, and an effort to integrate theory and practice in avalanche survival techniques.
```

## References

https://python.langchain.com/docs/tutorials/rag/
