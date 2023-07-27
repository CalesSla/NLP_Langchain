# Import dependencies
import os
import time
from dotenv import load_dotenv, find_dotenv
from helpers import load_document, load_from_wikipedia, chunk_data, print_embedding_cost, insert_or_fetch_embeddings, delete_pinecone_index, ask_and_get_answer
load_dotenv(find_dotenv(), override=True)  

# Testing functionality
# -------------------------------------------------------------------------
# data = load_document('Files/us_constitution.pdf')
# print(data[1].page_content)
# print('\n')
# print(data[10].metadata)
# print('\n')
# print(f'You have {len(data)} pages in your document')
# print(f'There are {len(data[20].page_content)} characters in the page')

# data = load_document('Files/the_great_gatsby.docx')
# print(data[0].page_content)


# External loader test
# data = load_from_wikipedia('GPT-4')
# print(data[0].page_content)

# with open('temp.txt', "w") as file:
#     file.write(str(data[0]))
 
# chunks = chunk_data(data)
# print(len(chunks))
# print(chunks[10].page_content)

# print_embedding_cost(chunks)


# delete_pinecone_index()

# index_name = 'askadocument'
# vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

# -------------------------------------------------------------------------

data = load_document('Files/us_constitution.pdf')

chunks = chunk_data(data)

delete_pinecone_index()

index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

question = 'What is the whole document about?'
answer = ask_and_get_answer(vector_store, question)
print(answer)

i = 1
print('Write Quit or Exit to quit.')
while True:
    question = input(f'Question #{i}: ')
    i = i+1
    if question.lower() == 'quit' or question.lower() == 'exit':
        print('Quitting...')
        time.sleep(1)
        break

    answer = ask_and_get_answer(vector_store, question)
    print(f'Answer: {answer}')
    print('-'*50)

delete_pinecone_index()

data = load_from_wikipedia('ChatGPT', 'ro')
chunks = chunk_data(data)
index_name = 'chatgpt'
vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

question = 'Explain InstructGPT'
answer = ask_and_get_answer(vector_store, question)
print(answer)


chat_history = []
question = 'How many amendments are in the US constitution?'
result, chat_history = ask_with_memory(vector_store, question, chat_history)
print(result['answer'])

question = 'Multiply that number by 2'
result, chat_history = ask_with_memory(vector_store, question, chat_history)
print(result['answer'])