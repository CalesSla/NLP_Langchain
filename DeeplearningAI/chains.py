import openai
import os
from dotenv import load_dotenv, find_dotenv
from helpers import get_completion
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.environ['OPENAI_API_KEY']


import pandas as pd
df = pd.read_csv('C:\\Users\\User\\Desktop\\Langchain\\DeeplearningAI\\data\\Data.csv')

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain = LLMChain(llm = llm, prompt = prompt, verbose = True)
product = 'Quality software'
# output = chain.run(product)
# print(output)


# Simple sequential chain
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9)

first_prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain_one  = LLMChain(llm = llm, prompt=first_prompt, verbose = True)
second_prompt = ChatPromptTemplate.from_template("Write a 20 words description for the following company: {company_name}")
chain_two = LLMChain(llm = llm, prompt = second_prompt, verbose = True)
overall_simple_chain = SimpleSequentialChain(chains = [chain_one, chain_two], verbose = True)
# print(overall_simple_chain.run(product))



# Sequential chain
from langchain.chains import SequentialChain
llm = ChatOpenAI(temperature=0.9)


first_prompt = ChatPromptTemplate.from_template("Translate the following review to English:\n\n {review}")
chain_one = LLMChain(llm = llm, prompt = first_prompt, verbose = True, output_key='English_review')

second_prompt = ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:\n\n {English_review}")
chain_two = LLMChain(llm=llm, prompt=second_prompt, verbose=True, output_key='summary')

third_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{review}")
chain_three = LLMChain(llm=llm, prompt=third_prompt, verbose=True, output_key = 'language')

fourth_prompt = ChatPromptTemplate.from_template("Write a follow up response to the following summary in the specified language with many swear words as you do not respect the customer:\n\n Summary: {summary}\n\n Language: {language}")
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, verbose=True, output_key = 'response')

overall_chain = SequentialChain(chains = [chain_one, chain_two, chain_three, chain_four], input_variables=['review'], output_variables = ['English_review', 'summary', 'response'], verbose = True)

review = """

Все кто хочет начать смотреть - идея изумительная, но 90% сериала это пустышные диалоги, излияние чувств то трансгендеров, то натуралов, то рассказы о прошлой жизни, которые впрочем имеют 0% реального веса для сюжета.

Сериал искусственно тянут, до одурения пробуя вытянуть из 1 сезона - 20 сезонов. Но в итоге уже на стадии 6 - 7 серии начался провал, к 9 серии дело дошло до того, что по животному делают, по капиталистически - 7 минут вступления, включая прошлые серии и заставку и потом реально 90% пустых диалогов. Мальчик ломается и признаётся девочке, девочка ломается и признаётся другому мальчику, итого две пары высказали нечто (я не слушал, сразу мотал) и в итоге 20 минут сожрано на признания парочек. На сюжет никакого влияния.

Ну конечно! Это ж так логично. Каждую ночь на город лезут монстры, свет берётся из проводов вне розеток, но мы должны слушать, что китаец думает о девочке и так далее. Да плевать всем на любовные треволнения! Это сериал о мистике, причём предельной, а нам пробуют вместо сюжета пихать пустые диалоги.
"""
print(overall_chain(review)['response'])



# Router chain
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0)
destination_chains = {}
for p_info in prompt_infos:
    name = p_info['name']
    prompt_template = p_info['prompt_template']
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm = llm, prompt = prompt, verbose = True)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = '\n'.join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt, verbose=True)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and  a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
print(chain.run("Why does every cell in our body contain DNA?"))