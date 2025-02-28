import os
import json
import requests

def extract_pdf_content_api(pdf_path, start_page, end_page, api_key='YOUR_API_KEY_HERE'):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'isOverlayRequired': False,
        'apikey': api_key,
        'language': 'eng'  # 필요 시 'kor' 또는 'kor+eng' 등으로 변경 가능
    }
    
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files, data=payload)
    
    result = response.json()
    parsed_results = result.get('ParsedResults', [])
    
    all_pages_result = []
    for page_num in range(start_page, min(end_page, len(parsed_results)) + 1):
        page_text = parsed_results[page_num - 1].get('ParsedText', '').strip()
        all_pages_result.append({
            "page_number": page_num,
            "content": page_text
        })
    
    return all_pages_result

if __name__ == "__main__":
    pdf_path = "/pdf/transformer.pdf"  # 분석할 PDF 파일 경로
    start_page = 1
    end_page = 3
    api_key = "YOUR_API_KEY_HERE"  # OCR.space API 키 (발급받은 키로 교체)
    
    result = extract_pdf_content_api(pdf_path, start_page, end_page, api_key)
    os.makedirs("json", exist_ok=True)
    with open("json/temp.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("완료")
