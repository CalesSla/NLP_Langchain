import openai
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# question = 'What does 13th ammendment imply?'

def us_constitution_helper(question):

    # Step 1. Load the document.
    from langchain.document_loaders import TextLoader
    loader = TextLoader("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\US_Constitution.txt")
    data = loader.load()

    # Step 2. Split the data into chunks.
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500)
    docs = text_splitter.split_documents(data)

    # Step 3. Embed docs and save to ChromaDB
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    embedding_function = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory='./us_bot_db')
    db.persist()
    docs = db.similarity_search(question)

    # Step 4. Context compression
    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm=llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db.as_retriever())
    compressed_docs = compression_retriever.get_relevant_documents(question)


    # Step 5. Answer the initial question.
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
    chat = ChatOpenAI(temperature=0.5)
    human_template = """Answer the question only using the information from the provided context.
    QUESTION: {question}
    CONTEXT: {context}"""
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt]).format_prompt(question=question, context=compressed_docs[0].page_content).to_messages()
    result = chat(chat_prompt)
    print(result.content)

us_constitution_helper('What does 1st ammendment imply?')