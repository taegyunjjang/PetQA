import os
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from time import sleep
import uuid
import click
from alive_progress import alive_bar

def get_question_img(soup):
    # 사진 첨부 여부 추출
    photo_attached = ["사진 없음"]
    # photo_tags = soup.find_all('img', class_='_rolling_0')
    photo_tags = soup.find_all('img', attrs={'class': '_rolling_0', 'alt': '첨부 이미지'})
    if not photo_tags: photo_tags = soup.find_all('span', class_='_waitingForReplaceImage')
    # print(photo_tags)
    # print('*'*20)
    # print(f'get_question_img: {photo_tags}')
    # print('*'*20)
    if len(photo_tags) > 0:
        photo_attached = []
        for photo_tag in photo_tags:
            try:
                image_url = photo_tag['src']
            except KeyError as e:
                image_url = photo_tag['data-thumb-image-src']

            try:
                image = requests.get(image_url)
            except:
                print('get_question_img: error raise')
                return [photo_tag['src'] for photo_tag in photo_tags]
            if image.status_code == 200:  
                image_uid = uuid.uuid4()
                file_name = f'/mnt/external_hdd2/petAI/kin_parsed_data0425/images/question/{str(image_uid)}.jpg' 
                with open(file_name, 'wb') as file: 
                    file.write(image.content)
                photo_attached.append(str(image_uid))
            else:
                photo_attached.append(image_url)
                
    return photo_attached

def get_answer_img(soup):
    # 사진 첨부 여부 추출
    photo_attached = ["사진 없음"]
    photo_tags = soup.find_all('img', class_='se-image-resource')
    # print('*'*20)
    # print(f'get_answer_img: {photo_tags}')
    # print('*'*20)
    if len(photo_tags) > 0:
        photo_attached = []
        for photo_tag in photo_tags:
            image_url = photo_tag['src']
            try:
                image = requests.get(image_url)
            except:
                print('get_answer_img: error raise')
                return [photo_tag['src'] for photo_tag in photo_tags]
            if image.status_code == 200:  # 요청 성공
                image_uid = uuid.uuid4()
                file_name = f'/mnt/external_hdd2/petAI/kin_parsed_data0425/images/answer/{str(image_uid)}.jpg'  # 저장할 파일 이름
                with open(file_name, 'wb') as file:  # 바이너리 모드로 저장
                    file.write(image.content)
                photo_attached.append(str(image_uid))
            else:
                photo_attached = ["사진 수집 실패"]
                return photo_attached
    return photo_attached

# HTML 파일에서 데이터를 추출하여 JSON 딕셔너리로 저장하는 함수
def parse_html_to_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 제목 추출
    title = soup.find('div', class_='endTitleSection').get_text(strip=True) if soup.find('div', class_='endTitleSection') else "제목 없음"
    
    # 질문 추출
    question = soup.find('div', class_='questionDetail').get_text(strip=True) if soup.find('div', class_='questionDetail') else "질문 없음"
    
    question_info_items = soup.find_all('span', class_='infoItem')
    question_date = None
    for item in question_info_items:
        if '작성일' in item.get_text():
            question_date = item.get_text(strip=True).replace('작성일', '', 1).strip()
            break

    # 태그 리스트 추출: 카테고리와 해시태그 분리
    category = []
    hashtags = []
    for tag in soup.select('div.tagList a.tag'):
        tag_text = tag.get_text(strip=True)
        if tag_text.startswith("#"):
            hashtags.append(tag_text)
        else:
            category.append(tag_text)
    tag_dict = {
        'category': category if category else ["카테고리 없음"],
        'hashtags': hashtags if hashtags else ["해시태그 없음"]
    }
    
    photo_attached = get_question_img(soup)

    # 동영상 첨부 여부 추출
    video_attached = "동영상 없음"
    video_tag = soup.find('div', class_='kin_movie_info')
    if video_tag:
        video_attached = "동영상 첨부"
    
    # 링크 추출
    url = soup.find('meta', property='og:url')
    link = url['content'] if url else "링크 없음"

    # 답변 수 추출
    answer_count_tag = soup.find('span', class_='_answerCount')
    answer_count = int(answer_count_tag.get_text(strip=True).replace(",", "").strip()) if answer_count_tag else 0

    final_return_dict = {
        '제목': title[2:],
        '본문': question,
        'question_video': video_attached,
        'tag_list': tag_dict,
        'link': link,
        'question_photo': photo_attached,
        'question_date': question_date
    }

    answers = []
    for i in range(1, answer_count + 1):
        # 각 답변의 ID (예: answer_1, answer_2 등)
        answer_container = soup.find('div', {'id': f'answer_{i}'})
        # print(answer_container)
        
        if answer_container:
            # 답변 텍스트 추출
            if len(answer_container.select('div.se-module-text span')) == 0:
                answer_detail_div = answer_container.find('div', class_='answerDetail _endContents _endContentsText')
                if answer_detail_div:
                    answer_text = answer_detail_div.get_text(separator=' ', strip=True)
                else:
                    continue # 작성자가 삭제한 답변                 
            else:
                answer_text = " ".join(
                    span.get_text(strip=True) 
                    for span in answer_container.select('div.se-module-text span')
                )

            answer_photo = get_answer_img(answer_container)

            # 채택 여부
            selected = False
            badge_tag = answer_container.find(
                'strong',
                class_='title',
                string=lambda s: s and '채택' in s
            )
            if badge_tag:
                selected = True

            answer_video = "동영상 없음"
            video_tag = answer_container.find('div', class_='se-component se-video se-l-default')
            if video_tag:
                answer_video = "동영상 첨부"

            # 답변자 정보 추출
            expert_badge = "전문가가 아님"
            expert_tag = answer_container.find('span', class_='badge expert_job')
            if expert_tag:
                expert_badge = expert_tag.get_text(strip=True)  # 뱃지에 표시된 직업 이름으로 저장

            eXpert_badge = "eXpert 아님"
            eXpert_tag = answer_container.find('span', class_='badge expert')
            if eXpert_tag:
                eXpert_badge = 'eXpert'  # naver eXpert 활동동

            badge = "뱃지 없음"
            badge_tag = answer_container.find('div', class_='badge_area')
            if badge_tag:
                level_badge = badge_tag.find('span', class_='badge')
                if level_badge:
                    badge = level_badge.get_text(strip=True)

            diligent_answerer = "열심 답변자 아님"
            diligent_badge = answer_container.find('span', class_='badge heavy_answer')
            if diligent_badge:
                diligent_answerer = "열심 답변자"

            career_list = "경력 정보 없음"
            career_tag = answer_container.find('div', class_='career_list')
            if career_tag:
                career_list = ', '.join([span.get_text(strip=True) for span in career_tag.find_all('span')])

            answer_date = answer_container.find('p', class_='answerDate').get_text().strip()

            answer_data = {
                '답변': answer_text,
                'expert_badge': expert_badge,
                'eXpert_badge': eXpert_badge,
                'badge': badge,
                'diligent_answerer': diligent_answerer,
                'career_list': career_list,
                'answer_photo': answer_photo,
                'answer_video': answer_video,
                'answer_date': answer_date,
                'selected': selected
            }
            
            answers.append(answer_data)
    final_return_dict['answers'] = answers
    return final_return_dict

# 중간 저장 함수 (기존 내용에 추가)
def save_filtered_json(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as json_file:
        for entry in data:
            json_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # print(f"Data appended to {output_file}")

# 폴더 내 모든 HTML 파일을 읽어 데이터를 JSON 형식으로 저장하는 함수
def folder_to_filtered_json(name, folder_path, save_interval=10):
    datas = {
        "개, 강아지": [],
        "고양이": [],
        "동물의료상담": [],
        "분양, 교배": [],
        "사료, 먹이": [],
        "훈련, 배변": [],
        "미용, 건강관리": [],
        "반려동물": [],
        "기타": []
    }
    # dog_data = []  # "개, 강아지" 데이터
    # cat_data = []  # "고양이" 데이터
    # medical_consultation_data = []  # "동물의료상담" 데이터
    # other_data = []  # 기타 데이터
    processed_files = 0

    parsing_path = '/mnt/external_hdd2/petAI/0609keyword_result/'
    # dog_result_path = f'{parsing_path}dog_data_experts.json'
    # cat_result_path = f'{parsing_path}cat_data_experts.json'
    # medical_consultation_result_path = f'{parsing_path}medical_consultation_data_experts.json'
    # other_result_path = f'{parsing_path}other_data_experts.json'

    result_paths = {
        "개, 강아지": f'{parsing_path}dog.json',
        "고양이": f'{parsing_path}cat.json',
        "동물의료상담": f'{parsing_path}medical_consultation.json',
        "분양, 교배": f'{parsing_path}mating.json',
        "사료, 먹이": f'{parsing_path}feedstuff.json',
        "훈련, 배변": f'{parsing_path}train.json',
        "미용, 건강관리": f'{parsing_path}beauty.json',
        "반려동물": f'{parsing_path}pet.json',
        "기타": f'{parsing_path}other.json'
    }
    with alive_bar(len(os.listdir(folder_path)), bar='scuba') as bar:
        for filename in os.listdir(folder_path):
            bar()
        # for filename in tqdm(os.listdir(folder_path), desc=f"{folder_path} parsing..."):
            if filename.endswith(".txt"):
                html_path = os.path.join(folder_path, filename)
                
                # 파일 내용 읽기
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    entries = parse_html_to_data(html_content)
                    entries["html_path"] = html_path
                    if entries['tag_list']['category'][0] in datas.keys():
                        datas[entries['tag_list']['category'][0]].append(entries)
                    else:
                        datas["기타"].append(entries)

                    processed_files += 1

                if processed_files % save_interval == 0:
                    for key in datas.keys():
                        if datas[key]:
                            save_filtered_json(datas[key], result_paths[key])
                            datas[key] = [] # 저장 후 초기화

    print(f"Intermediate save after processing {processed_files} files.")

    for key in datas.keys():
        if datas[key]:
            save_filtered_json(datas[key], result_paths[key])

    # 모든 파일 처리 후 최종 저장
    # if dog_data:
    #     save_filtered_json(dog_data, dog_result_path)
    # if cat_data:
    #     save_filtered_json(cat_data, cat_result_path)
    # if medical_consultation_data:
    #     save_filtered_json(medical_consultation_data, medical_consultation_result_path)
    # if other_data:
    #     save_filtered_json(other_data, other_result_path)

    print(f"Final data saved after processing all files.")

@click.command()
@click.option('--month', type=int, required=True, help='1~12')
def main(month):
    base_folder = '/mnt/external_hdd2/petAI/kin_keyword_remove_duplicates'
    for year in [2021, 2022, 2023, 2024]:
        for pet in ['dog', 'cat']:
        # for pet in ['cat']:
            final_path = f"{base_folder}/{year}/{month}/{pet}/"
            print(f'folder path: {final_path}')
            folder_to_filtered_json("name", final_path)

if __name__ == "__main__":
    main()