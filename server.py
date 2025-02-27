from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import magic
from main import main

app = FastAPI()

# 업로드 폴더 설정
BASE_DIR = Path("./")
UPLOAD_DIRS = {
    "pdf": BASE_DIR / "PDF",
    "mp3": BASE_DIR / "MP3"
}

# 폴더 생성
for dir_path in UPLOAD_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

@app.post("/quizzes")
async def upload_file(
    file: UploadFile = File(...),
    type: str = Form(...),  # 문제 형식 (CHOICE or SUBJECT)
    start: int = Form(...),
    end: int = Form(...),
    keyword: str = Form(...)
):
    
    # Quiz 타입 설정
    quiz_type = type

    # MIME 타입 확인
    mime = magic.Magic(mime=True)
    file_mime_type = mime.from_buffer(await file.read(2048))  # 파일의 처음 2KB로 MIME 검사

    # MIME 타입을 기반으로 파일 타입 결정
    if file_mime_type == "application/pdf":
        file_type = "pdf"
    elif file_mime_type in ["audio/mpeg", "audio/mp3"]:
        file_type = "mp3"
    else:
        return {"error": "Unsupported file type"}

    # 저장할 폴더 선택
    file_path = UPLOAD_DIRS[file_type] / file.filename

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Main function 실행
    json_res = main(file_type, str(file_path), "json/input.json", quiz_type, keyword, start, end)

    # 처리 후 파일 삭제
    file_path.unlink(missing_ok=True)

    return json_res

@app.get("/", summary="root EndPoint", description="Check API is Working on Server.")
async def read_root():
    return {"name": "quiz_whale"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
