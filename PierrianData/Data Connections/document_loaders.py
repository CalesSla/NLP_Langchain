from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")



# CSV loader
loader = CSVLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\penguins.csv")
data = loader.load()
# print(data)
# print(type(data[0]))


# HTML loader
from langchain.document_loaders import BSHTMLLoader
loader  = BSHTMLLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\some_website.html")
# data = loader.load()
# print(data)
 

# PDF loader
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\SomeReport.pdf")
# data = loader.load()
# print(data[0].page_content)


# Hacker News loader and comment summarization
from langchain.document_loaders import HNLoader
loader = HNLoader("https://news.ycombinator.com/item?id=36697119")
data = loader.load()
print(data[0].metadata)

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

human_template = 'Please give me a short summary of the following HackerNews comment:\n{comment}'
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

model = ChatOpenAI()
result = model(chat_prompt.format_prompt(comment=data[0].page_content).to_messages())
print(result.content)