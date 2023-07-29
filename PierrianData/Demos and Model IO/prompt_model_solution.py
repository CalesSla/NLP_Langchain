import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage



def travel_idea(interest, budget):
    chat = ChatOpenAI(temperature=0.5, max_tokens=500)

    system_template = "You are Mike, an experienced and polite travel advisor who is knowledgeable in {interest}. You don't ask any extra questions from the customer."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = 'I like {interest} and I have a budget of {budget}. What travel plan can you recommend me'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    final_prompt = chat_prompt.format_prompt(interest = interest, budget = budget).to_messages()

    result = chat(final_prompt)
    print(result.content)

travel_idea('diving', '2000 Euro')

# llm = OpenAI(temperature=0.5, max_tokens=200)
# print(llm('Tell me 3 insteresting facts about Moldova'))