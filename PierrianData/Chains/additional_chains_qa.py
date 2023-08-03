from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
embedding_functio = OpenAIEmbeddings()
db = Chroma(persist_directory='C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\vector_db',  embedding_function=embedding_functio)

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type='stuff')

question = "What is the 15th ammendment?"
docs = db.similarity_search(question)

# result = chain.run(input_documents=docs, question = question)
# print(result)


chain = load_qa_with_sources_chain(llm, chain_type='stuff')
result = chain.run(input_documents=docs, question = question)
print(result)


