from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI()
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(llm=llm, memory=memory)
conversation.predict(input='Hello how are you?')
conversation.predict(input='Tell me a math fact.')
conversation.predict(input='Tell me a fact about Mars.')
print(memory.buffer)
print(memory.load_memory_variables({}))