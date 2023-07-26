import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
load_dotenv(find_dotenv(), override=True)  

template = """
    You are an experienced virologist.
    Write a few sentences about the following virus {virus} in {language}
"""

prompt = PromptTemplate(input_variables = ['virus', 'language'], template = template)

llm = OpenAI(model_name = 'text-davinci-003', temperature = 0.7, max_tokens = 512)

output = llm(prompt.format(virus = 'ebola', language = 'Romanian'))
print(output)