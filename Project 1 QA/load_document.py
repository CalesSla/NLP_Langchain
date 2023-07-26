# Helper function to load content of a PDF file
def load_document(file):
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