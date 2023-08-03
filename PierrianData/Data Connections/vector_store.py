from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")



import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


# General workflow:
# 1. Load document
# 2. Split into chunks
# 3. Use embedding model to embed chunks into vectors
# 4. Get vector chunks and save them to vector store
# 5. Define a query and do similarity search

loader = TextLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\US_Constitution.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 500)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function, persist_directory="C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\vector_db")
db.persist()

db_new_connection = Chroma(persist_directory="C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\vector_db", embedding_function=embedding_function)

new_doc = 'What did FDR say about the cost of food law?'
similar_docs = db.similarity_search(new_doc)



loader = TextLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\Lincoln_State_of_Union_1862.txt")
documents = loader.load()

docs = text_splitter.split_documents(documents)
db = Chroma.from_documents(docs, embedding_function, persist_directory="C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\vector_db")

docs = db.similarity_search('lincoln')

retriever = db.as_retriever()
print(len(retriever.get_relevant_documents('cost of food law')))


