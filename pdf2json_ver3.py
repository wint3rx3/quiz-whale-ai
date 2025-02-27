import os
import json
from pdfminer.high_level import extract_text

def extract_pdf_content(pdf_path, start_page, end_page):
    all_pages_result = []
    # PDFMiner는 0부터 시작하는 페이지 인덱스를 사용하므로 변환합니다.
    for page_num in range(start_page, end_page + 1):
        # extract_text 함수에 page_numbers 파라미터로 특정 페이지(0-indexed)를 지정
        text = extract_text(pdf_path, page_numbers=[page_num - 1])
        all_pages_result.append({
            "page_number": page_num,
            "content": text.strip()  # 양쪽 공백 제거
        })
    return all_pages_result

if __name__ == "__main__":
    pdf_path = "/pdf/transformer.pdf"  # 분석할 PDF 파일 경로
    start_page = 1
    end_page = 3
    result = extract_pdf_content(pdf_path, start_page, end_page)
    # 결과를 JSON 파일로 저장 (utf-8 인코딩, 한글 출력 유지)
    with open("json/temp.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("완료")
