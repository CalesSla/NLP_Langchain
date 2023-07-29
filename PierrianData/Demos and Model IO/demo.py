import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.Completion.create(model = 'text-davinci-003', prompt = "Give me two reasons to learn OpenAI with Python", max_tokens = 300)
# print(response.choices[0].text)



# Using LLMs with LangChain
from langchain.llms import OpenAI
llm = OpenAI()
# print(llm('Here is a fun fact about Pluto:'))
# result = llm.generate(['Here is a fact about Plute', 'Here is a fact about Mars'])
# # print(result.schema())
# for i in result.generations:
#     print(i[0].text)
# print(result.generations)



# Chat models with LangChain
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()

from langchain.schema import AIMessage, HumanMessage, SystemMessage

result = chat([SystemMessage(content = 'You are a very rude teenager who just wants to party and not answer questions'),
               HumanMessage(content = 'Tell me a fact about Pluto')])

# print()
# print(result.content)
# print()

# result = chat.generate([
#     [SystemMessage(content = 'You are a very rude teenager who just wants to party and not answer questions'),
#                HumanMessage(content = 'Tell me a fact about Pluto')],
#     [SystemMessage(content = 'You are a friendly assistant'),
#                HumanMessage(content = 'Tell me a fact about Pluto')]
# ])

# for i in result.generations:
#     print(i[0].text)
#     print()



# Extra parameters
# result =  chat([SystemMessage(content = 'You are a friendly assistant'), HumanMessage(content = 'Tell me a fact about Pluto')],
#                temperature = 0, presence_penalty = 2, max_tokens = 20)
# print(result.content)




# Caching answers
import langchain
from time import perf_counter
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()

start = perf_counter()
print(llm.predict('Tell me a fact about Mars'))
end = perf_counter()
print(end - start)
print()
start = perf_counter()
print(llm.predict('Tell me a fact about Mars'))
end = perf_counter()
print(end - start)