from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


beginner_template = """
You are a physics teacher who is really focused on beginners and explaining concepts in simple to understand terms.
You assume no prior knowledge. Here is your question:\n{input}
"""
expert_template = """
You are a physics professor who explains physics topics to advanced audience members. You can assume anyone you answer
has a PhD in Physics. Here is your question:\n{input}
"""

prompt_infos = [
    {'name': 'beginner physics',
     'description': "Answers basic physics question",
     'template': beginner_template},

     {'name': 'advanced physics',
     'description': 'Answers advanced physics questions',
     'template': expert_template}
]

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI()
destination_chains = {}
for p in prompt_infos:
    name = p['name']
    prompt_template = p['template']
    prompt = ChatPromptTemplate.from_template(template = prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_prompt = ChatPromptTemplate.from_template('{input}')
default_chain = LLMChain(llm = llm, prompt = default_prompt)

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
destinations = [f"{p['name']}: {p['description']}"  for p in prompt_infos]
destination_str = "\n".join(destinations)
print(destination_str)


from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations = destination_str)
print(router_template)

router_prompt = PromptTemplate(template=router_template,
                               input_variables=['input'],
                               output_parser = RouterOutputParser())

from langchain.chains.router import MultiPromptChain

router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True)

result = chain.run('Please explain Feyman Diagrams')