import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import openai
import pprint
load_dotenv(find_dotenv(), override=True)  

llm = ChatOpenAI(model_name = 'gpt-4-0314', temperature=0.5)

template = """
    You are an experienced virologist.
    Write a few sentences about the following virus {virus} in {language}
"""

prompt = PromptTemplate(input_variables=['virus', 'language'], template=template)

chain = LLMChain(llm=llm, prompt=prompt)

output = chain.run({'virus': 'HSV', 'language':'Russian'})
print(output)



