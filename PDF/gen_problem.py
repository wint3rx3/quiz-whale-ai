import os
import openai
import json
from dotenv import load_dotenv

# Initialize
load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Load JSON data
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to generate JSON-structured problems
def generate_problem(data, problem_type, key_point, problem_id=0):
    page = data["page_number"]
    content = " ".join(data["content"])  # Merge content list into a single string

    if problem_type == "multiple_choice":
        prompt = (
            f"Generate a multiple-choice question with 5 choices based on the following content and key points:\n\n"
            f"Content: {content}\n"
            f"Key points: {key_point}\n\n"
            f"Return the problem in this JSON format:\n"
            f'{{"id": {problem_id}, "title": "{key_point}", "problem": "<question_text>", '
            f'"choices": {{"A": "<choice 1>", "B": "<choice 2>", "C": "<choice 3>", "D": "<choice 4>", "E": "<choice 5>"}}, '
            f'"answer": "<correct choice (A/B/C/D/E)>"}
            f'"explanation": "explanation for problem and correct answer"}'
        )
    else:  #O/X problem
        prompt = (
            f"Generate a True/False (O/X) question based on the following content and key points:\n\n"
            f"Content: {content}\n"
            f"Key points: {key_point}\n\n"
            f"Return the problem in this JSON format:\n"
            f'{{"id": {problem_id}, "title": "{key_point}", "problem": "<statement_text> (O/X)", "answer": "<O/X>"}}, '
            f'without explanation.'
        )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = response.choices[0].message.content
    # print(f"Raw GPT Response for problem ID {problem_id}:\n{response_text}")

    try:
        problem_json = json.loads(response_text)
        return problem_json
    except json.JSONDecodeError:
        print(f"Error decoding JSON for problem ID {problem_id}")
        return None

# -------------------------------------------------
if __name__ == "__main__":
    # Example usage
    generated_problems = []
    pages = load_json("temp.json")

    for i, page in enumerate(pages):
        generated_problem = generate_problem(page, "multiple_choice", "Not given", problem_id=i + 1)
        if generated_problem:
            generated_problems.append(generated_problem)
        else:
            exit()

    # Save the generated problem to a JSON file
    with open("generated_problems.json", "w", encoding="utf-8") as f:
        json.dump(generated_problems, f, ensure_ascii=False, indent=4)

    print("Generated problems saved to generated_problems.json")
