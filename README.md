# ​ RAG-Powered Document Chatbot

Upload PDFs. Ask questions. Get grounded answers with citations powered by FAISS search and local LLMs.

---

##  Features
-  Upload and parse PDF documents (_PyMuPDF_)
-  Text chunking and embedding via `sentence-transformers`
-  Vector search with FAISS
-  Chat UI in Streamlit with a polished hero header & dark-mode styling
-  Retrieval from local LLM models via Ollama (_e.g. `mistral`, `llama3`_)
-  Citations displayed with similarity scores

---

##  Setup & Usage


git clone https://github.com/LOHITHK22/RAG-Powered-Document-Chatbot.git
cd RAG-Powered-Document-Chatbot
python -m venv venv
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
pip install -r requirements.txt


##Start Ollama (local model support)

ollama serve
ollama pull mistral

##Run the app

streamlit run app.py

##Dependencies :-

streamlit
numpy
faiss-cpu
pymupdf
sentence-transformers
requests

License
MIT License — feel free to use, modify, and share.