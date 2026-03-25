"""
Generate 1000 Indian names using a free HuggingFace LLM


Author: Sandesh Suman
"""

import torch
from transformers import pipeline

OUTPUT_FILE = "TrainingNames.txt"
TOTAL_NAMES = 1000


def main():
    print("Loading model (first time may take time)...")

    # Small free model 
    generator = pipeline(
        "text-generation",
        model="distilgpt2"   # lightweight model
    )

    prompt = (
        "List of Indian names:\n"
        "Aarav\nVivaan\nAnanya\nDiya\nKabir\nIshaan\n"
    )

    names_set = set()

    print("Generating names...")

    while len(names_set) < TOTAL_NAMES:
        outputs = generator(
            prompt,
            max_length=200,
            num_return_sequences=3,
            temperature=0.9,
            do_sample=True
        )

        for out in outputs:
            text = out["generated_text"]

            # Split lines
            lines = text.split("\n")

            for name in lines:
                name = name.strip().lower()

                # Clean conditions
                if name.isalpha() and 3 <= len(name) <= 12:
                    names_set.add(name)

        print(f"Collected: {len(names_set)} names")

    # Keep only 1000
    final_names = list(names_set)[:TOTAL_NAMES]

    # Save file
    with open(OUTPUT_FILE, "w") as f:
        for name in final_names:
            f.write(name + "\n")

    print(f"\n Saved {len(final_names)} names to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()