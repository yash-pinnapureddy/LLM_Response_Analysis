import argparse
import json
from confidence_engine import LLMConfidenceEngine

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)

args = parser.parse_args()

engine = LLMConfidenceEngine()
result = engine.analyze(args.prompt)

print(json.dumps(result, indent=2))