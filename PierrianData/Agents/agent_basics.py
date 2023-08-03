from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(tools=tools, llm=llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
result = agent.run("What is 13456 times 56213")


print(type(result))