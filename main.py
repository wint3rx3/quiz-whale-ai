import os
import json

from gen_problem import load_json
from gen_problem import generate_problem
from pdf2json import extract_pdf_content
from speech_to_json import transcribe_audio


def main(file_format, file_path, json_path, problem_format, key_points, start_page, end_page):
    # file format: "pdf" or "mp3"
    # file_path: "pdf/test.pdf", "mp3/test.mp3", ...
    # json_path: "json/test.json", ...
    # problem_format: "choice" or "subject"
    # key_points: a string
    # start_page, end_page: pdf range

    if file_format == "pdf":
        json_data = extract_pdf_content(file_path, start_page, end_page)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    else:
        json_data = transcribe_audio(file_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    # Input file to JSON file
    # --------------------------------------------------------

    generated_problems = []
    pages = load_json(json_path)

    for i, page in enumerate(pages):
        generated_problem = generate_problem(page, problem_format, key_points, problem_id=i + 1)
        if generated_problem:
            generated_problems.append(generated_problem)
        else:
            print("Could not generate problem!")
            exit()

    # Save the generated problem to a JSON file
    print("Generated problems saved to output.json")
    return json.dump(generated_problems, f, ensure_ascii=False, indent=4)
