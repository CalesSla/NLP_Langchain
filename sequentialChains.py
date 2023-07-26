import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import openai
import pprint
load_dotenv(find_dotenv(), override=True)  

llm1 = OpenAI(model_name = 'text-davinci-003', temperature=0.5, max_tokens=1024)
prompt1 = PromptTemplate(input_variables=['concept'], template="""You are an experienced scientist and python programmer. Write a function that implements the concept of {concept}.""")
chain1 = LLMChain(llm=llm1, prompt=prompt1)

llm2 = ChatOpenAI(model_name = 'gpt-4', temperature=1.2, max_tokens = 100)
prompt2 = PromptTemplate(input_variables=['function'], template="""
    Given the python function {function} describe it as detailed as possible""")
chain2 = LLMChain(llm=llm2, prompt=prompt2)

overall_chain = SimpleSequentialChain(chains = [chain1, chain2], verbose = True)
output = overall_chain.run('linear regression')

