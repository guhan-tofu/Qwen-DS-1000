import os
import json
import sys
import argparse

from agents import Agent, Runner, function_tool
from generate_data import pandas_prompt, matplotlib_prompt, numpy_prompt, sklearn_prompt, scipy_prompt, pytorch_prompt, tensorflow_prompt
from pydantic import BaseModel

class SampleFormat(BaseModel):
    question: str 
    answer: str


async def generate_samples(num=3, out_file="generated_samples_4.1mini.jsonl", model="gpt-4.1-mini", temperature=0.7, max_tokens=800):
    agent = Agent(
        name="Data Science Problem Generator",
        instructions= ("Generate a new data science coding problem in the specified format."
                       """ You are to use the pydantic model provided to you to give structured output. It has two fields: "question" and "answer". The "question" field should contain the Problem and Setup code. The "answer" field should contain the Answer."""),
        model=model,
        tools=[],
        output_type=SampleFormat,
    )

    results = []
    for _ in range(num):
        response = await Runner.run(agent, tensorflow_prompt)
        output = response.final_output
        # convert pydantic BaseModel (or similar) into primitives for JSON serialization
        if hasattr(output, "dict") and callable(getattr(output, "dict")):
            entry = output.dict()
        else:
            entry = {"text": str(output)}
        print(entry)
        results.append(entry)

    with open(out_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(results)




import asyncio

def main():
    parser = argparse.ArgumentParser(description="Generate samples using Agent and the prompt from generate_data.py")
    parser.add_argument("-n", "--num", type=int, default=50, help="number of samples to generate")
    parser.add_argument("-o", "--out", default="generated_samples_tensorflow.jsonl", help="output JSONL file")
    parser.add_argument("--model", default="gpt-4.1-mini", help="model to call")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    try:
        count = asyncio.run(generate_samples(num=args.num, out_file=args.out, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens))
        print(f"Wrote {count} samples to {args.out}")
    except Exception as e:
        print("Error generating samples:\n", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
