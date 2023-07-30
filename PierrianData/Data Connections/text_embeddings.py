from langchain.document_loaders import CSVLoader
import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")



from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
text = "This is some normal text string that I want to embed as a vector"
embedded_text = embeddings.embed_query(text)
print(embedded_text)


from langchain.document_loaders import CSVLoader
loader = CSVLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\penguins.csv")
data = loader.load()
embedded_docs = embeddings.embed_documents([text.page_content for text in data])

