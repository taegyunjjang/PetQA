import re
from tqdm import tqdm
from pypdf import PdfReader
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import load_environment



def clean_text(text):
    cleaned_text = text
    
    # DOI 및 저작권 정보 제거
    header_pattern = r'Introduction to Diseases, Diagnosis, and Management of Dogs and Cats\..*?All rights reserved\.'
    cleaned_text = re.sub(header_pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # 빈 페이지 제거
    blank_page_pattern = r'this\s+page\s+intentionally\s+left\s+blank'
    cleaned_text = re.sub(blank_page_pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # references 제거
    cleaned_text = re.split(r'^\s*References\s*$', cleaned_text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    
    return cleaned_text.strip()

    
env = load_environment()

reader = PdfReader(env["textbook_path"])
all_text = []

CHAPTER_START = 33
INDEX_START = 641

pages = reader.pages
for i, page in enumerate(tqdm(pages, desc="Processing pages")):
    page_number = i + 1
    
    if page_number < CHAPTER_START or page_number >= INDEX_START:
        continue
    
    text = page.extract_text()
    if not text or not text.strip():
        continue
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        continue
    
    all_text.append(cleaned_text)
        
combined_text = "\n\n".join(all_text)  # 페이지 구분

output_path = "./textbook.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"Total text length: {len(combined_text)}")