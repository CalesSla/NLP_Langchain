import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
load_dotenv(find_dotenv(), override=True)

chat = ChatOpenAI(model_name='gpt-4', temperature = 0.5, max_tokens = 1024)
messages = [
    SystemMessage(content='You are a person who uses a lot of swear words and respond only in Russian'),
    HumanMessage(content='Explain quantum mechanics in one sentence'),
]

output = chat(messages)
print(output.content)