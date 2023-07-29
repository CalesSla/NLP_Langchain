import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat = ChatOpenAI()

# Using f-string literals
# planet = 'Venus'
# print(llm(f'Here is a fun fact about {planet}:'))


# Prompt templates
from langchain import PromptTemplate

# no_input_prompt = PromptTemplate(input_variables=[], template = 'Tell me a fact')
# print(llm(no_input_prompt.format()))


# single_input_prompt = PromptTemplate(input_variables=['topic'], template = 'Tell me a fact about {topic}')
# print(llm(single_input_prompt.format(topic = 'Mars')))

# multi_input_prompt = PromptTemplate(input_variables=['topic', 'level'], template = 'Tell me a fact about {topic} for a {level} student')
# print(llm(multi_input_prompt.format(topic = 'the ocean', level = 'PhD')))




# Chat prompt templates
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_template = "You are an AI recipy assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = '{recipe_request}'
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# print(system_message_prompt.input_variables)
# print(human_message_prompt.input_variables)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# print(chat_prompt.input_variables)
prompt = chat_prompt.format_prompt(cooking_time = '60 min', recipe_request = 'Quick Snack', dietary_preference = 'Vegan').to_messages()
result = chat(prompt)
print(result.content)