import os
import numpy as np
import json
from pdf2image import convert_from_path
import easyocr
from PIL import Image

def extract_pdf_content(pdf_path, start_page, end_page, dpi=300):
    # EasyOCR Reader 초기화 (CPU 사용)
    reader = easyocr.Reader(['en', 'ko'], gpu=False)
    
    # PDF를 PIL 이미지 리스트로 변환 (dpi 설정)
    images = convert_from_path(pdf_path, dpi=dpi)
    selected_images = images[start_page - 1: end_page]

    all_pages_result = []
    for page_index, pil_image in enumerate(selected_images, start=start_page):
        # 전체 페이지에 대해 OCR 수행
        ocr_results = reader.readtext(np.array(pil_image))
        # OCR 결과에서 인식된 텍스트만 추출
        text_segments = [item[1] for item in ocr_results]
        page_content = "\n".join(text_segments)
        all_pages_result.append({"page_number": page_index, "content": page_content})
    
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
