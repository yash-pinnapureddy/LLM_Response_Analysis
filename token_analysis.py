import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def estimate_cost(tokens, price_per_1k=0.0005):
    return (tokens / 1000) * price_per_1k


if __name__ == "__main__":
    text = "This is a sample sentence to analyze token usage."
    
    token_count = count_tokens(text)
    cost = estimate_cost(token_count)

    print(f"Tokens: {token_count}")
    print(f"Estimated cost: ${cost:.6f}")
