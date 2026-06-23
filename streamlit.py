import streamlit as st
from embeddings import get_embedding, cosine_similarity
from token_analysis import count_tokens

st.title("LLM Analytics Playground")

text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")

if st.button("Analyze"):
    if text1 and text2:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)

        similarity = cosine_similarity(emb1, emb2)

        tokens1 = count_tokens(text1)
        tokens2 = count_tokens(text2)

        st.subheader("Results")
        st.write(f"Similarity: {similarity:.4f}")
        st.write(f"Tokens (Text1): {tokens1}")
        st.write(f"Tokens (Text2): {tokens2}")