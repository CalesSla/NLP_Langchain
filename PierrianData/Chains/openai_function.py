from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo-0613')

class Scientist():
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

json_schema = {"title": 'Scientist',
               'description': "Information about a famous scientist",
               'type': 'object',
               'properties': {
                   'first_name': {'title': 'First Name',
                                  'description': 'First name of scientist',
                                  'type': 'string'},
                   'last_name' : {'title': 'Last Name',
                                  'description': 'Last name of scientist',
                                  'type': 'string'}
                },
                'required': ['first_name', 'last_name']
               }

template = 'Name a famous {country} scientist'

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.prompts import ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_template(template)

chain = create_structured_output_chain(json_schema, llm, chat_prompt, verbose = True)
result = chain.run(country = 'German')

albert = Scientist(result['first_name'], result['last_name'])
