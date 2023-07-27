def ask_and_get_answer(vector_store, question):
    """Ask a question and get an answer based on thecontext in vector store"""
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(question)
    return answer


def ask_with_memory(vector_store, question, chat_history = []):
    """Ask a question and get an answer based on the context in vector store WITH MEMORY"""
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature = 1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})

    chat_history.append((question, result['answer']))

    return result, chat_history
