import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  

def chunk_data(data, chunk_size=2000):
    """Helper function to chunk data into smaller pieces"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks


def insert_or_fetch_embeddings(index_name, chunks):
    """Create a pinecone index and insert embeddings or fetch existing embeddings into a vector store"""
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Done')
    else:
        print(f'Creating index {index_name}....', end = '')
        pinecone.create_index(index_name, dimension = 1536, metric='cosine')
        pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Done')
    return vector_store


def delete_pinecone_index(index_name='all'):
    """Delete a pinecone index"""
    import pinecone
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all pinecone indexes')
        for index in indexes:
            pinecone.delete_index(index)
        print('Done')
    else:
        print(f'Deleteing index {index_name}')


