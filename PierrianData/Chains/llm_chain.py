from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

human_template = "Make up a funny company name for a company that makes: {product}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

chat = ChatOpenAI()



from langchain.chains import LLMChain
chain = LLMChain(llm=chat, prompt=chat_prompt)
result = chain.run(product='Computers')
print(result)