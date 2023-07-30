import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, DatetimeOutputParser
import datetime

class HistoryQuiz():
    
    def create_history_question(self,topic):
        '''
        This method should output a historical question about the topic that has a date as the correct answer.
        For example:
            "On what date did World War 2 end?"
        '''
        chat = ChatOpenAI(temperature=2)

        system_template = """You are a system that builds historical quiz questions about {topic}. 
        The answer to the question must be a full date and not just a year, but the date MUST NOT be part of the question itself.
        When you are asked you only provide a question and nothing more.
        The question must start with the words: "What date..."
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{topic}"
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        prompt = chat_prompt.format_prompt(topic = topic).to_messages()
        result = chat(prompt)
        # print(f'Generated the following question: {result.content}')

        return result.content
    
    def gen_AI_answer(self, question):
        chat = ChatOpenAI(temperature=0)
        output_parser = DatetimeOutputParser()

        system_template = """You are an expert historian.
        Answer the question by providing the correct date in your answer and nothing more.
        """
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = "{question}\n{format_instructions}"
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        prompt = chat_prompt.format_prompt(question=question, format_instructions=output_parser.get_format_instructions()).to_messages()
        result = chat(prompt)

        parsed_result = output_parser.parse(result.content)
        # print(f'Generated the following answer: {parsed_result}')

        return parsed_result
    
    def get_user_answer(self, question):
        
        print(f"Answer the following question\n{question}")
        year = int(input("Enter the year: "))
        month = int(input("Enter the month: "))
        day = int(input("Enter the day: "))
        date_time = datetime.datetime(year, month, day)
        return date_time

    def check_user_answer(self, user_answer, ai_answer):
        time_difference = ai_answer - user_answer
        # print(time_difference.days)
        if time_difference.days != 0:
            print(f"Wrong answer! The correct answer is: {ai_answer}")
        else:
            print("Correct answer!")

a = HistoryQuiz()
question = a.create_history_question('Moldova')
AI_answer = a.gen_AI_answer(question)
user_answer = a.get_user_answer(question)
dif = a.check_user_answer(user_answer, AI_answer)