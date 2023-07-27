def load_document(file):
    """Helper funcion to load a document from a file"""
    import os
    name, ext = os.path.splitext(file)

    if ext == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading document {file}....')
        loader = PyPDFLoader(file)
    elif ext == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading document {file}....')
        loader = Docx2txtLoader(file)
    else:
        print('Document format not supported')
        return None
    data = loader.load()
    return data


def load_from_wikipedia(query, lang='en', load_max_docs=2):
    """Helper function to load a document from wikipedia"""
    from langchain.document_loaders import WikipediaLoader
    print(f'Loading document {query} from wikipedia....')
    loader = WikipediaLoader(query, lang=lang, load_max_docs=load_max_docs, doc_content_chars_max=20000)
    data = loader.load()
    return data