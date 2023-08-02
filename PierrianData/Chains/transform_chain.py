from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


yelp_review = open('C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\yelp_review.txt').read()

from langchain.chains import TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def transformer_function(inputs: dict) -> dict:
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output': lower_case_text}

transform_chain = TransformChain(input_variables=['text'], output_variables=['output'], transform=transformer_function)

template = 'Create a one sentence summary of this review:\n{review}'

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(template)
summary_chain = LLMChain(llm=llm, prompt=prompt, output_key='review_summary')

sequential_chain = SimpleSequentialChain(chains=[transform_chain, summary_chain], verbose = True)
result = sequential_chain(yelp_review)

