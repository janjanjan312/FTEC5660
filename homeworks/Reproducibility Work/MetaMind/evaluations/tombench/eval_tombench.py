import json
import os
import re
from tqdm import tqdm

from main import MetamindApplication

def main():
    app = MetamindApplication()

    data_dir = "evaluations/tombench"
    files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]

    total_correct = 0
    total_examples = 0

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                
                story = example.get("STORY", "")
                question = example.get("QUESTION", "")
                option_a = example.get("OPTION-A", "")
                option_b = example.get("OPTION-B", "")
                option_c = example.get("OPTION-C", "")
                option_d = example.get("OPTION-D", "")
                answer = example.get("ANSWER", "")

                options_str = f"A: {option_a}\nB: {option_b}\nC: {option_c}\nD: {option_d}"

                user_utterance = f"{story}\n\n{question}\n\nOptions:\n{options_str}\n\nPlease select the correct answer by outputting only the letter (A, B, C, or D)."

                result = app.process_user_input(user_utterance, conversation_context=[])
                response = result["final_response"]

                # Simple parsing: find the first A/B/C/D in the response
                match = re.search(r"\b[A-D]\b", response)
                predicted = match.group(0) if match else None

                if predicted == answer:
                    total_correct += 1

                total_examples += 1

    accuracy = (total_correct / total_examples) * 100 if total_examples > 0 else 0
    print(f"Total examples: {total_examples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
