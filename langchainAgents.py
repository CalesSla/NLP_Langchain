import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import openai
import pprint
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
load_dotenv(find_dotenv(), override=True)  

llm = OpenAI(temperature = 0)
agent_executor = create_python_agent(llm = llm, tool = PythonREPLTool(), verbose = True)

output = agent_executor.run('Calculate the square root of the factorial of 20 and display it with 4 decimal points' )
print(output)

output = agent_executor.run('what is the answer to 5.1 ** 7.3?')
print(output)