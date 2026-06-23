# LLM Confidence Analyzer

## Features
- Confidence Score (logprobs)
- Token Analysis
- Embeddings
- Retry handling
- Best-of-N selection
- Streamlit UI + CLI

## Run

```bash
pip install -r requirements.txt
cp .env.example .env
```

## CLI
```bash
python main.py --prompt "Explain AI"
```

## UI
```bash
streamlit run app.py
```

## Formula
confidence = mean(exp(logprob))
