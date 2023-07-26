import os
import random
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import openai
import pprint
import pinecone

load_dotenv(find_dotenv(), override=True)  
pinecone.init(api_key = os.environ.get('PINECONE_API_KEY'), environment = os.environ.get('PINECONE_ENV'))
print(pinecone.info.version())

print(pinecone.list_indexes())
print('\n')

index_name = 'langchain-pinecone'
if index_name not in pinecone.list_indexes():
    print(f'Creating index {index_name}....')
    pinecone.create_index(index_name, dimension = 1536, metric = 'cosine', pods = 1, pod_type = 'p1.x2')
    print('Done')
else:
    print(f'Index {index_name} already exists')

print(pinecone.describe_index(index_name))

index = pinecone.Index(index_name = index_name)
print(index.describe_index_stats())


vectors = [[random.random() for _ in range(1536)] for v in range(5)]
ids = list('abcde')

index.upsert(vectors=zip(ids, vectors))

# updating a vector
index.upsert(vectors=[('c', [0.3] * 1536)])

# fetching a vector by id
index.fetch(ids=['c'])

# deleting vectors by id
# index.delete(ids = ['b', 'c'])

# deleting all vectors
# index.delete(delete_all=True)

# querying
queries = [[random.random() for _ in range(1536)] for v in range(2)]

index.query(queries=queries, top_k=3, include_values=False)