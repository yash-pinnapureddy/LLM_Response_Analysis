from openai import OpenAI
import numpy as np
from typing import Dict, List

client = OpenAI()


class LLMConfidenceEngine:

    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    # ==============================
    # 1. GET RESPONSE WITH LOGPROBS
    # ==============================
    def generate_response(self, prompt: str, temperature: float = 0) -> Dict:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=5,
            temperature=temperature
        )
        return response


    # ==============================
    # 2. EXTRACT TOKEN DATA SAFELY
    # ==============================
    def extract_token_data(self, response) -> List[Dict]:
        try:
            tokens_data = response.choices[0].logprobs.content
            return tokens_data if tokens_data else []
        except Exception:
            return []


    # ==============================
    # 3. TOKEN-LEVEL ANALYSIS
    # ==============================
    def token_level_analysis(self, tokens_data: List[Dict]) -> List[Dict]:
        results = []

        for t in tokens_data:
            prob = float(np.exp(t.logprob))

            results.append({
                "token": t.token,
                "logprob": float(t.logprob),
                "probability": prob,
                "top_alternatives": [
                    {
                        "token": alt.token,
                        "logprob": float(alt.logprob),
                        "probability": float(np.exp(alt.logprob))
                    }
                    for alt in (t.top_logprobs or [])
                ]
            })

        return results


    # ==============================
    # 4. RESPONSE-LEVEL CONFIDENCE
    # ==============================
    def response_confidence(self, tokens_data: List[Dict]) -> Dict:

        if not tokens_data:
            return {
                "confidence_score": 0.0,
                "avg_logprob": None,
                "perplexity": None
            }

        logprobs = np.array([t.logprob for t in tokens_data])
        probs = np.exp(logprobs)

        avg_prob = float(np.mean(probs))
        avg_logprob = float(np.mean(logprobs))
        perplexity = float(np.exp(-avg_logprob))

        # FULL sequence probability (can underflow)
        sequence_probability = float(np.exp(np.sum(logprobs)))

        return {
            "confidence_score": avg_prob,           # Most useful metric
            "avg_logprob": avg_logprob,
            "perplexity": perplexity,
            "sequence_probability": sequence_probability
        }


    # ==============================
    # 5. LOW-CONFIDENCE DETECTION
    # ==============================
    def detect_low_confidence_tokens(self, tokens_data: List[Dict], threshold=0.4) -> List[str]:
        low_tokens = []

        for t in tokens_data:
            prob = np.exp(t.logprob)
            if prob < threshold:
                low_tokens.append(t.token)

        return low_tokens


    # ==============================
    # 6. OVERALL QUALITY SIGNAL
    # ==============================
    def compute_quality_label(self, confidence_score: float) -> str:
        if confidence_score > 0.8:
            return "HIGH"
        elif confidence_score > 0.6:
            return "MEDIUM"
        else:
            return "LOW"


    # ==============================
    # 7. FULL PIPELINE
    # ==============================
    def analyze(self, prompt: str) -> Dict:

        response = self.generate_response(prompt)
        output_text = response.choices[0].message.content

        tokens_data = self.extract_token_data(response)

        token_analysis = self.token_level_analysis(tokens_data)
        confidence_metrics = self.response_confidence(tokens_data)
        low_conf_tokens = self.detect_low_confidence_tokens(tokens_data)

        quality = self.compute_quality_label(
            confidence_metrics["confidence_score"]
        )

        return {
            "prompt": prompt,
            "output_text": output_text,
            "confidence_metrics": confidence_metrics,
            "quality": quality,
            "low_confidence_tokens": low_conf_tokens,
            "token_analysis": token_analysis
        }


# ==============================
# ✅ RUN EXAMPLE
# ==============================
if __name__ == "__main__":

    engine = LLMConfidenceEngine()

    prompt = "Explain quantum computing in simple terms"

    result = engine.analyze(prompt)

    print("\n🧠 OUTPUT:")
    print(result["output_text"])

    print("\n📊 CONFIDENCE METRICS:")
    print(result["confidence_metrics"])

    print("\n⚠️ LOW CONFIDENCE TOKENS:")
    print(result["low_confidence_tokens"])

    print("\n🏷 QUALITY:")
    print(result["quality"])
