from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import fitz  # PyMuPDF
from main import main

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # 업로드 폴더 생성

@app.post("/quizzes")
async def upload_file(
    file: UploadFile = File(...),
    type: str = Form(...),  # 파일 타입 (pdf 또는 audio)
    start: int = Form(...),
    end: int = Form(...),
    keyword: str = Form(...)
):
    file_path = UPLOAD_DIR / file.filename

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Main function
    if type == "pdf" or type == "mp3":
        json_res = main(type, str(file_path), "json/input.json", "choice", "Not given", 1, 3)
    else:
        return {"error": "Invalid file type"}

    # 파일 삭제
    file_path.unlink(missing_ok=True)

    # json 반환
    return json_res
