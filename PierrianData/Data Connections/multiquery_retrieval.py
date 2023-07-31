from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.document_loaders import WikipediaLoader
loader = WikipediaLoader(query = 'MKUltra')
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500)
docs = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
embedding_function = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embedding_function, persist_directory='./some_new_mkultra')
db.persist()


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
question = "When was this declassified?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)



import logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(unique_docs[0].page_content)
