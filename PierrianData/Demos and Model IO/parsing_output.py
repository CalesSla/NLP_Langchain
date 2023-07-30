import os
import openai
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
model = ChatOpenAI()



from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
# print(output_parser.get_format_instructions())

human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
model_request = chat_prompt.format_prompt(request='give me 5 characteristics of dogs', format_instructions=output_parser.get_format_instructions()).to_messages()

# result = model(model_request)
# print(type(result.content))
# parsed_result = output_parser.parse(result.content)
# print(parsed_result)



from langchain.output_parsers import DatetimeOutputParser, OutputFixingParser
output_parser = DatetimeOutputParser()
# print(output_parser.get_format_instructions())

human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# system_template = "You always reply to questions only in datetime patterns."
# system_prompt = SystemMessagePromptTemplate.from_template(system_template)

chat_prompt = ChatPromptTemplate.from_messages([
    # system_prompt, 
    human_prompt
                                                ])
model_request = chat_prompt.format_prompt(request = "What date was the 13th Amendment ratified in the US?", format_instructions = output_parser.get_format_instructions()).to_messages()

# result = model(model_request)
# print(result.content)
# # print(output_parser.parse(result.content))

# misformatted = result.content

# new_parser = OutputFixingParser.from_llm(parser = output_parser, llm = model)
# print(new_parser.parse(misformatted))








# Pydantic parser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Scientist(BaseModel):
    name: str = Field(descrition = 'Name of a scientist')
    discoveries: list = Field(description='List of discoveries made by the scientist')

parser = PydanticOutputParser(pydantic_object=Scientist)
# print(parser.get_format_instructions()) 

human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

request = chat_prompt.format_prompt(request = "Tell me about a famous scientist", format_instructions = parser.get_format_instructions()).to_messages()

result = model(request, temperature = 0)
print(result)