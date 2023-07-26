# Import dependencies
import os
from dotenv import load_dotenv, find_dotenv
from load_document import load_document, load_from_wikipedia

# Testing the loader function
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


