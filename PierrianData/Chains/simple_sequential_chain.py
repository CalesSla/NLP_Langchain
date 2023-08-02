from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
llm = ChatOpenAI()

template = "Give me a simple bullet point outline for a blog post on {topic}"
first_prompt = ChatPromptTemplate.from_template(template)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

template2 = "Write me a blog post using this outline {outline}"
second_prompt = ChatPromptTemplate.from_template(template2)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

full_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
result = full_chain.run('Cheesecake')
print(result)