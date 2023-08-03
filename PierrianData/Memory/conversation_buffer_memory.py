from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose = True)
conversation.predict(input = 'Hello, nice to meet you.')
conversation.predict(input = 'Tell me about an interesting Physics fact.')
print(memory.buffer)

history = memory.load_memory_variables({})

import pickle
pickled_string = pickle.dumps(conversation.memory)
with open('convo_memory.pkl', 'wb') as f:
    f.write(pickled_string)


new_memory_loaded = open('convo_memory.pkl', 'rb').read()
llm = ChatOpenAI()
reloaded_conversation = ConversationChain(llm=llm, memory=pickle.loads(new_memory_loaded))
print(reloaded_conversation.memory.buffer)