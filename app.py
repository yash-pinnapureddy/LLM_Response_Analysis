import streamlit as st
from confidence_engine import LLMConfidenceEngine, best_of_n
from token_analysis import count_tokens

engine = LLMConfidenceEngine()

st.title("🧠 LLM Confidence Analyzer")

prompt = st.text_area("Enter your prompt")

if st.button("Analyze"):
    result = engine.analyze(prompt)

    st.subheader("Response")
    st.write(result["output"])

    st.subheader("Confidence")
    st.json(result["metrics"])
    st.write(result["quality"])

    tokens = count_tokens(result["output"])
    st.write(f"Tokens: {tokens}")

if st.button("Best of 3"):
    result = best_of_n(engine, prompt)

    st.subheader("Best Response")
    st.write(result["output"])
    st.write(result["metrics"])