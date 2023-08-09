from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history')
llm = ChatOpenAI(temperature=0)
tools = load_tools(['llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
result = agent.run(input="What are some good Thai food recipes?")
print(result)
result2 = agent.run("Which one of those dishes tends to be the spiceiest?")
print(result2)
result3 = agent.run('Give me a grocery shopping list for that dish')
print(result3)