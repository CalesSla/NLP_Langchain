from langchain import PromptTemplate
template = "Tell me a fact about {planet}"

prompt = PromptTemplate(template = template, input_variables=['planet'])
prompt.save("myprompt.json")

from langchain.prompts import load_prompt
loaded_prompt = load_prompt("myprompt.json") 