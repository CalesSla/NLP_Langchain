from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# result = agent.run("Give me a summary of Mihail Ivanov's linkedin profile. He is from Moldova Chisinau. .net developer")
# print(result)


from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents.agent_toolkits import create_python_agent

llm = ChatOpenAI(temperature=0)
agent = create_python_agent(tool=PythonREPLTool(), llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
python_list = [1,3,2,4,1,2,3,4,1,1,2,3,5,10]
result = agent.run(f"""Sort this python list: {python_list}""")