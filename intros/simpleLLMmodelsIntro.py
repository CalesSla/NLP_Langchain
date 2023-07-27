import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
load_dotenv(find_dotenv(), override=True)

llm = OpenAI(model_name = 'text-davinci-003', temperature=0.7, max_tokens=512)

# prompt = 'explain quantum mechanics in one sentence'
# output = llm(prompt)
# print(output)
# print('\n')
# print(llm.get_num_tokens(prompt))

# output = [llm.generate(['... is the capital of France', 'What is the formula for the area of a circle?'])]
# print(output[0].generations[0].text.strip())
# print(len(output[0].generations))
# print('\n')

output = llm.generate(['Write an original tagline for a burger restaurant'] * 3)
print(output)

for i in output.generations:
    print(i[0].text.strip())