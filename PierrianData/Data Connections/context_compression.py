from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.vectorstores import Chroma
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

embedding_function = OpenAIEmbeddings()
db_connection = Chroma(persist_directory='./some_new_mkultra', embedding_function=embedding_function)

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db_connection.as_retriever())

docs = db_connection.similarity_search('When was this declassified?')
print(docs[0])

compressed_docs = compression_retriever.get_relevant_documents('When was this declassified?')