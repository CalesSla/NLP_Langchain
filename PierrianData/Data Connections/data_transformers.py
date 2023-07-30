from langchain.text_splitter import CharacterTextSplitter

with open("C:\\Users\\User\\Desktop\\Langchain\\PierrianData\\some_data\\FDR_State_of_Union_1944.txt") as file:
    speech_text = file.read()

text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size = 1000)
texts = text_splitter.create_documents([speech_text])



text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500)
texts = text_splitter.split_text(speech_text)
