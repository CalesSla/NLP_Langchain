import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_cost import print_embedding_cost
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


load_dotenv(find_dotenv(), override=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
with open('Files/churchill_speech.txt', 'r') as file:
    churchcill_speech = file.read()

chunks = text_splitter.create_documents([churchcill_speech])

print(chunks[10].page_content)
print('\n')

print_embedding_cost(chunks)
print('\n')

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query(chunks[0].page_content)

print(len(vector))

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

# deleting all pinecone indexes
# indexes = pinecone.list_indexes()
# for index in indexes:
#     print('Deleting all pinecone indexes')
#     pinecone.delete_index(index)

index_name = 'churchill-speech'
if index_name not in pinecone.list_indexes():
    print(f'Creating index {index_name}....')
    pinecone.create_index(index_name, dimension=1536, metric='cosine')
    print('Done')

vector_store = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)

# Asking questions (similarity search)
# query = 'Where should we fight?'
query = 'Who was the king of Belgium at that time?'

result = vector_store.similarity_search(query=query, k = 3)
for i in result:
    print(i.page_content)
    print('-' * 50)

llm = ChatOpenAI(model='gpt-4', temperature=1, max_tokens=100)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs = {'k': 3})

chain = RetrievalQA.from_chain_type(llm = llm, chain_type='stuff', retriever=retriever)

answer = chain.run(query)
print(answer)