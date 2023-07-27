import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

load_dotenv(find_dotenv(), override=True)

text = """The British Isles consists of two large islands and many small ones. The largest island is Great Britain and the second largest is Ireland. 
Scotland, England and Wales are the countries which make up Great Britain. Ireland consists of the Republic of Ireland and Northern Ireland. 
Great Britain and Northern Ireland, together, make up the United Kingdom.
The UK is governed from London, but Scotland, Wales and N. Ireland also have their own governments for domestic affairs. 
The government in London deals with external affairs and defence. The Republic of Ireland is an independent country.
Great Britain is about 1000 km from north to south and about 500 km from east to west at the widest part. The landscape of the British Isles is varied. 
Scotland has most mountains, but there are also mountains in northern Wales. Ben Nevis (1,343 m) in Scotland is the highest mountain in the British Isles. 
Northern England has some hills but they are not as high as those in Scotland and Wales. Southern England is relatively flat and suitable for agriculture. 
Ireland is commonly called the Emerald Isle due to the green countryside
"""

messages = [SystemMessage(content='You are an expert copywriter with expertise in summarizing documents'),
            HumanMessage(content=f'Please provide a short and concise summary of the following text:\n TEXT: {text}')]

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
print(llm.get_num_tokens(text))

summary_output = llm(messages)
print(summary_output.content)


# Summarizing using prompt templates
template = """Write a concise and short summary of the following text:
TEXT: {text}
Translate the summary to {language}"""

prompt = PromptTemplate(input_variables=['text', 'language'], template=template)
print(llm.get_num_tokens(prompt.format(text = text, language = 'English')))

chain = LLMChain(llm=llm, prompt=prompt)

summary = chain.run({'text': text, 'language': 'Russian'})
print(summary)


# Summarizing using StuffDocumentChain
with open('Files/sj.txt', encoding = 'utf-8') as f:
    text = f.read()

docs = [Document(page_content=text)]
template = """Write a concise and short summary of the following text:
TEXT: {text}"""
prompt = PromptTemplate(input_variables=['text'], template=template)

chain = load_summarize_chain(llm, chain_type='stuff', prompt=prompt, verbose = True)
output_summary = chain.run(docs)
print(output_summary)


# Summarizing large documents using map-reduce
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
chunks = text_splitter.create_documents([text])
print(len(chunks))

chain = load_summarize_chain(llm, chain_type = 'map_reduce', verbose = True)
output_summary = chain.run(chunks)
print(output_summary)

print(chain.llm_chain.prompt.template)


# map-reduce with custom prompts
map_prompt = """Write a short and concise summary of the following:
Text: {text}
CONCISE SUMMARY: 
"""
map_prompt_template = PromptTemplate(input_variables=['text'], template=map_prompt)
combine_prompt =    """
Write a concise summary of the following text that coverts the key points.
Add a title to the summary.
Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
Text: {text}
"""
combine_prompt_template = PromptTemplate(input_variables=['text'], template=combine_prompt)
summary_chain = load_summarize_chain(llm = llm, chain_type = 'map_reduce', map_prompt = map_prompt_template, combine_prompt = combine_prompt_template, verbose = True)

output = summary_chain.run(chunks)]
print(output)



# Summarizing using the refine chain
