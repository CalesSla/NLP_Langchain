import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0, max_tokens=500)

system_template = "You are a helpful legal assistant that translates complex legal terms into plain and understandable terms."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

legal_text = "The provisions herein shall be severable, and if any provision or portion thereof is deemed invalid, illegal, or unenforceable by a court of competent jurisdiction, the remaining provisions or portions thereof shall remain in full force and effect to the maximum extent permitted by law.\"\n"
example_input_one = HumanMessagePromptTemplate.from_template(legal_text)

plain_text = "The rules in this agreement can be separated."
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

human_template = "{legal_text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_input_one, example_output_one, human_message_prompt])
# print(chat_prompt.input_variables)
example_legal_text = "The grantor, being the fee simple owner of the real property herein described, conveys and warrants to the grantee, his heirs and assigns, all of the grantor's right, title, and interest in and to the said property, subject to all existing encumbrances, liens, and easements, as recorded in the official records of the county, and any applicable covenants, conditions, and restrictions affecting the property, in consideration of the sum of [purchase price] paid by the grantee.\"\n"
request = chat_prompt.format_prompt(legal_text = example_legal_text).to_messages()
# for i in range(len(request)):
#     print(str(type(request[i])).split('.')[-1][:-2] + ': ' + request[i].content)

result = chat(request)
print(result.content)