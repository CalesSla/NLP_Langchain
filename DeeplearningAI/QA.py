import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

file = 'C:\\Users\\User\\Desktop\\Langchain\\DeeplearningAI\\data\\OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query)
print(display(Markdown(response)))