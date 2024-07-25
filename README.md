# fashion-rag
_Tutorial Followed: https://www.youtube.com/watch?v=de6_BjEVWZo_

The first step into the RAG world via fashion context. The "assistant" answers the user's question using the data from the fashion PDF file.

I used a model from HuggingFace and an open-source vector store.

I created a RAG chain using the following:
- For embedding, the model: sentence-transformers/all-MiniLM-L6-v2
- For VectorDB: ChromaDB (`pip install chromadb
- As the LLM: mistralai/Mixtral-8x7B-Instruct-v0.1

**Short and simple,**
I create the vector database from the PDF file containing fashion data and use the LLM to query it and answer a question.
I kept it simple, but you can extend it, correct it, and even add a UI (with Streamlit like in the video).

**To start:**
> (Optional) You can create and activate a virtual environment first.
1. install the requirements: `pip install -r requirements.txt`
2. Create a `.env` file at the project's root.
3. Place your `HUGGINGFACEHUB_API_TOKEN` inside.

You're good to go from there.
Good luck!
