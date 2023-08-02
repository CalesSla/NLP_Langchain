from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

model = ChatOpenAI(model='gpt-4')
result = model([HumanMessage(content='What is 17 raised to the power of 11')])
# print(result)

from langchain import LLMMathChain
llm_math_model = LLMMathChain.from_llm(model)
result = llm_math_model('What is 17 raised to the power of 11?')
print(result)