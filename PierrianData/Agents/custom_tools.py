from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.agents import load_tools, initialize_agent, AgentType, tool
from langchain.chat_models import ChatOpenAI

@tool
def coolest_guy(text: str) -> str:
    """Returns the name of the coolest person in the universe.
       Expects an input of an empty string '' and returns the
       name of the coolest person in the universe.
       """
    return "Slava Calestru"

llm = ChatOpenAI(temperature=0)
tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
tools = tools + [coolest_guy]
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run('Who is the coolest person in the universe?')
print(result)


from datetime import datetime

@tool
def get_current_time(text):
    """Returns the current time. Use this for any questions
    regarding the current time. Input is an empty string '' and the current time
    is returned in a string format. Only use this function for the current time. 
    Other time related questions should use another tool."""
    return str(datetime.now())

agent = initialize_agent(tools+[get_current_time], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run('What is the current time?')
print(result)

result = agent.run("What time did Pearl Harbor attack happen?")
print(result)