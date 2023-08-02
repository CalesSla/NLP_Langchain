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

