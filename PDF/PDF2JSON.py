import os
import torch
import numpy as np
import json
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from PIL import Image

# OCR 및 캡션 관련 라이브러리 임포트
import easyocr
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration

# pix2tex 임포트 (수식 인식을 위해)
from pix2tex.cli import LatexOCR

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_doclayout_model():
    model_filepath = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    return YOLOv10(model_filepath)

def detect_regions(model, image, imgsz=1024, conf_threshold=0.2):
    image_array = np.array(image)
    results = model.predict(image_array, imgsz=imgsz, conf=conf_threshold, device=DEVICE)
    return results

def convert_pdf_to_images(pdf_path, dpi=200):
    # PDF를 PIL 이미지 객체 리스트로 변환 (디스크 저장 없이)
    return convert_from_path(pdf_path, dpi=dpi)

def crop_region(image, bbox):
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))

def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    return processor, model

def image_caption(cropped_img, caption_processor, caption_model):
    if cropped_img.mode != "RGB":
        cropped_img = cropped_img.convert("RGB")
    inputs = caption_processor(cropped_img, return_tensors="pt").to(DEVICE)
    output = caption_model.generate(**inputs)
    return caption_processor.decode(output[0], skip_special_tokens=True)

def pix2tex_formula(cropped_img, formula_ocr):
    return formula_ocr(cropped_img)

def extract_pdf_content(pdf_path, start_page, end_page, dpi=300):
    """
    pdf_path: PDF 파일 경로
    start_page: 시작 페이지 번호 (1-indexed)
    end_page: 끝 페이지 번호 (1-indexed, 포함)
    
    반환 예시:
      [
         {"page_number": 1, "content": "(plain text) ...\n(title) ...\n(...)" },
         {"page_number": 2, "content": "(plain text) ...\n(...)" },
         ...
      ]
    """
    # 모델 및 필요한 객체 로드
    doclayout_model = load_doclayout_model()
    reader = easyocr.Reader(['en', 'ko'], gpu=torch.cuda.is_available())
    caption_processor, caption_model = load_captioning_model()
    formula_ocr = LatexOCR()
    
    # PDF를 메모리 내에서 이미지 객체로 변환
    images = convert_pdf_to_images(pdf_path, dpi=dpi)
    
    # 페이지 범위 조정 (리스트는 0-indexed)
    selected_images = images[start_page - 1: end_page]
    
    all_pages_result = []
    for page_index, pil_image in enumerate(selected_images, start=start_page):
        results = detect_regions(doclayout_model, pil_image)
        text_segments = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            class_name = doclayout_model.names[cls_id]
            xyxy = box.xyxy[0]
            bbox = list(map(int, xyxy))
            
            # 영역별 처리: 표, 일반 텍스트, 그림, 수식 등
            if class_name in ["plain text", "title", "table_caption", "formula_caption", "table_footnote", "table"]:
                cropped_img = crop_region(pil_image, bbox)
                ocr_result = reader.readtext(np.array(cropped_img))
                recognized_text = "\n".join([item[1] for item in ocr_result])
            elif class_name == "figure":
                cropped_img = crop_region(pil_image, bbox)
                recognized_text = image_caption(cropped_img, caption_processor, caption_model)
            elif class_name == "isolate_formula":
                cropped_img = crop_region(pil_image, bbox)
                recognized_text = pix2tex_formula(cropped_img, formula_ocr)
            else:
                recognized_text = ""
            
            # 만약 인식된 텍스트가 있다면, (클래스) 접두어를 붙여서 추가
            if recognized_text:
                segment = f"({class_name}) {recognized_text}"
                text_segments.append(segment)
        
        # 모든 영역을 하나의 줄글(문자열)로 병합
        page_content = "\n".join(text_segments)
        page_info = {"page_number": page_index, "content": page_content}
        all_pages_result.append(page_info)
    
    return all_pages_result

# 예시: 함수 호출 후 반환된 결과를 JSON 파일로 저장하는 코드
if __name__ == "__main__":
    pdf_path = "/content/transformer.pdf"  # 분석할 PDF 파일 경로
    start_page = 1
    end_page = 3
    result = extract_pdf_content(pdf_path, start_page, end_page)
    json_output_path = "temp.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("완료")