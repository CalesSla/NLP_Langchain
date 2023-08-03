from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()
history.add_user_message("Hello my name is Slava")
history.add_ai_message('Hi, my name is Chatgpt')

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import HumanMessage

llm=ChatOpenAI()

result = llm(history.messages + [HumanMessage(content='What is my name?')])