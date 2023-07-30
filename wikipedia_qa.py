import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader

def answer_question(person_name, question):
    loader = WikipediaLoader(query=person_name, load_max_docs=1)
    data = loader.load()
    context = data[0].page_content


    chat = ChatOpenAI(temperature=0)

    system_template = """Answer the {question} about {person_name}. Only base your answer on the {context}"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{question}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format_prompt(question=question, person_name=person_name, context=context).to_messages()
    result = chat(chat_prompt)
    print(result.content)
    return result.content

answer_question(person_name = 'Albert Einstein', question = 'Tell me an interesting fact about him')