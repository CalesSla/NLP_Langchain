def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total tokens: {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens / 1000 * 0.0004:.6f}')