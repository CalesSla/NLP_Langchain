from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


spanish_email = open('C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\spanish_customer_email.txt').read()

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import SequentialChain, LLMChain
# from langchain.prompts import ChatPromptTemplate

# llm = ChatOpenAI()

# template1 = "Given the email identify and return only the language of it without any extra words:\n{email}"
# prompt1 = ChatPromptTemplate.from_template(template=template1)
# chain1 = LLMChain(llm=llm, prompt=prompt1, output_key='language')

# template2 = "Given the email translate it from {language} to {translation_language}:\n{email}"
# prompt2 = ChatPromptTemplate.from_template(template=template2)
# chain2 = LLMChain(llm=llm, prompt=prompt2, output_key='translated_email')

# template3 = "Create a short summary of this email:\n{translated_email}"
# prompt3 = ChatPromptTemplate.from_template(template=template3)
# chain3 = LLMChain(llm=llm, prompt=prompt3, output_key='email_summary')

# full_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['email', 'translation_language'], output_variables=['language', 'translated_email', 'email_summary'])

# results = full_chain({'email': spanish_email, 'translation_language': 'English'})




from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI()

template1 = "Given the email text identify the language. Return only the language and no extra words:\n{email}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key='language')

template2 = "Given the email translate it from {language} to {translation_language}:\n{email}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key='translated_email')

template3 = "Given the email text provide a short summary of it:\n{translated_email}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key='email_summary')

final_chain = SequentialChain(chains=[chain1, chain2], input_variables=['email', 'translation_language'], output_variables=['language', 'translated_email', 'email_summary'])
results = final_chain({'email': spanish_email, 'translation_language': 'English'})