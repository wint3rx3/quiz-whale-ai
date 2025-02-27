import os
import numpy as np
import json
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from PIL import Image

# OCR 및 캡션 관련 라이브러리 임포트
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from pix2tex.cli import LatexOCR


def extract_pdf_content(pdf_path, start_page, end_page, dpi=300):
    # 모델들을 한 번에 로드
    doclayout_model = YOLOv10(hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    ))
    reader = easyocr.Reader(['en', 'ko'])
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    formula_ocr = LatexOCR()
    
    # PDF를 메모리 내 이미지 객체 리스트로 변환
    images = convert_from_path(pdf_path, dpi=dpi)
    selected_images = images[start_page - 1: end_page]

    all_pages_result = []
    for page_index, pil_image in enumerate(selected_images, start=start_page):
        # 문서 레이아웃 영역 검출
        results = doclayout_model.predict(np.array(pil_image), imgsz=1024, conf=0.2, device=DEVICE)
        text_segments = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            class_name = doclayout_model.names[cls_id]
            bbox = list(map(int, box.xyxy[0]))
            cropped_img = pil_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            recognized_text = ""
            if class_name in ["plain text", "title", "table_caption", "formula_caption", "table_footnote", "table"]:
                ocr_result = reader.readtext(np.array(cropped_img))
                recognized_text = "\n".join([item[1] for item in ocr_result])
            elif class_name == "figure":
                if cropped_img.mode != "RGB":
                    cropped_img = cropped_img.convert("RGB")
                inputs = caption_processor(cropped_img, return_tensors="pt").to(DEVICE)
                output = caption_model.generate(**inputs)
                recognized_text = caption_processor.decode(output[0], skip_special_tokens=True)
            elif class_name == "isolate_formula":
                recognized_text = LatexOCR(cropped_img)
            
            if recognized_text:
                text_segments.append(f"({class_name}) {recognized_text}")
        
        page_content = "\n".join(text_segments)
        all_pages_result.append({"page_number": page_index, "content": page_content})
    
    return all_pages_result

# 예시: 결과를 JSON 파일로 저장
if __name__ == "__main__":
    pdf_path = "/pdf/transformer.pdf"  # 분석할 PDF 파일 경로
    start_page = 1
    end_page = 3
    result = extract_pdf_content(pdf_path, start_page, end_page)
    with open("json/temp.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("완료")
