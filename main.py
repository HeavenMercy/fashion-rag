from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

with open('data/prompt_template.txt', 'r') as f:
    prompt_template = f.read()

prompt = PromptTemplate(template=prompt_template, input_variables=['question'])

pdfloader = PyPDFLoader('data/fashion_data.pdf')
documents = pdfloader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_db = Chroma(embedding_function=embed_model, persist_directory='db')

model = HuggingFaceEndpoint(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
                            temperature=1, max_new_tokens=1024)

rag_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=vector_db.as_retriever(top_k=3),
    chain_type_kwargs={'prompt': prompt})


def get_answer(question):
    # last_in_prompt = 'Answer:'

    result = rag_chain.invoke({'query': question})
    result = result['result']
    # end_prompt = result.index(last_in_prompt) + len(last_in_prompt)

    # result = result[end_prompt:].strip()
    return result


if __name__ == '__main__':
    print(get_answer("What can you recommend that is made off of cashmere?"))
