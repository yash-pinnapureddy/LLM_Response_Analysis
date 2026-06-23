from openai import OpenAI

client = OpenAI()

def get_logprobs(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5
    )
    
    tokens = response.choices[0].logprobs.content

    for t in tokens:
        print(f"Token: {t.token}")
        print(f"LogProb: {t.logprob}")
        print("Top alternatives:")
        for alt in t.top_logprobs:
            print(f"   {alt.token}: {alt.logprob}")
        print("-" * 40)


if __name__ == "__main__":
    get_logprobs("The future of AI is")