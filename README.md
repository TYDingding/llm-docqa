# PDF Q&A Chatbot with LangChain + OpenAI / Ollama

This project is a chatbot that can answer questions about local PDF using **LangChain**, **OpenAI GPT (API version) / Ollama (local version)**, and **FAISS**. It demonstrates a Retrieval-Augmented Generation (RAG) workflow, where your question is answered based on the contents of a document.

## Features

- Upload and query local PDF files
- OpenAI API version:
  - Embeds documents using OpenAI Embeddings
  - Generates answers using ChatGPT API (gpt-4.1-nano)
- Ollama local version:
  - Embeds documents using Ollama Embeddings (nomic-embed-text)
  - Generates answers using Yollama customized from gemma3:1b
- Retrieves relevant chunks using FAISS vector store
- Use [tqdm](https://github.com/tqdm/tqdm) to visualize the process

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

3. To use OpenAI API version (main.py), add your OpenAI API Key

   Create a .env file:

   ```bash
   cp .env.example .env
   ```

   Edit .env:

   ```ini
   OPENAI_API_KEY=your-key-here
   ```

4. Add your PDF
   Place your PDF file inside the pdfs/ folder and rename it sample.pdf (or change the filename in main.py or main_local.py).
5. To use Ollama version (main_local.py), set your Ollama model name in fuction `built_qa_chain(vectorstore)`
6. To custoize a prompt based on Ollama, Please refer [Custoize a prompt](https://github.com/ollama/ollama?tab=readme-ov-file#customize-a-prompt). In this project, I used both a Yollama_Modelfile (based on gemma3:1b) and a PromptTemplate in code to define a system prompt, but after experimentation and research I found that system prompts behave differently depending on whether they are embedded via

   1. Modelfile (static). In CLI, the Ollama works well with Modelfile. However, the same approach does not show customized system prompt in code with LangChain.
   2. Injected at runtime with PromptTemplate. Strangely this approach make approach i work.

   If you discover why certain models ignore one approach, or how to reliably enforce behavior—I’d love to hear from you!

## Run The App

OpenAI version:

```bash
python main.py
```

Ollama version:

```bash
python main_local.py
```

Example:

```vbnet
You: What is this pdf about?

DocQA: This PDF appears to be a scholarly article from the proceedings of the 2008 International Snow Science Workshop. It discusses avalanche survival strategies, particularly focusing on different parts of a flowing avalanche and how individuals can improve their chances of survival. The paper includes a practitioner's perspective, theoretical insights into avalanche dynamics, case studies of avalanche survivors, and an effort to integrate theory and practice in avalanche survival techniques.
```

## References

https://python.langchain.com/docs/tutorials/rag/
https://ollama.com/library/gemma3
https://ollama.com/library/nomic-embed-text
https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html
