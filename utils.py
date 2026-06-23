import time
import logging

logging.basicConfig(level=logging.INFO)


def retry(func, retries=3, delay=1):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Retry {i+1} failed: {e}")
            time.sleep(delay)
    raise Exception("Max retries exceeded")


def apply_guardrail(confidence):
    if confidence < 0.5:
        return "⚠️ Low confidence"
    elif confidence < 0.7:
        return "⚠️ Medium confidence"
    return "✅ High confidence"
