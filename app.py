from flask import Flask, render_template, request
import json
import os
from datetime import datetime
from our_model.emotion_model import improved_analyzer 

app = Flask(__name__)

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'diaries.json')

# --- 날짜 변환 ---
def format_english_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d, %Y')
    except ValueError:
        return date_str 

# --- 저장 함수 ---
def save_diary(data):
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        diaries = []
    diaries.append(data)
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(diaries, f, ensure_ascii=False, indent=4)
    print(f"✅ Saved: {data['date']}")

# --- 불러오기 함수 ---
def load_diaries():
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            diaries = json.load(f)
            diaries.reverse() 
            return diaries
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# --- [NEW] 모든 등장인물 이름 모으기 ---
def get_all_people():
    diaries = load_diaries()
    people_set = set()
    for diary in diaries:
        # 일기에 'people' 정보가 있으면 이름만 뽑아서 저장
        if 'people' in diary:
            for p in diary['people']:
                people_set.add(p['name'])
    # 알파벳 순으로 정렬해서 리스트로 반환
    return sorted(list(people_set))

# ================= 라우팅 =================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    # 1. 검색 조건 받기 (감정, 날짜, [NEW] 사람)
    filter_emotion = request.args.get('emotion')
    filter_date = request.args.get('date')
    filter_person = request.args.get('person') # 사람 조건 추가

    all_diaries = load_diaries()
    
    # [NEW] 검색창에 띄워줄 '모든 인물 명단' 가져오기
    all_people_list = get_all_people()

    filtered_diaries = []
    
    for diary in all_diaries:
        # 1. 감정 필터링
        if filter_emotion and filter_emotion != "All" and diary['emotion'] != filter_emotion:
            continue
        
        # 2. 날짜 필터링
        if filter_date:
            english_filter_date = format_english_date(filter_date)
            if diary['date'] != english_filter_date:
                continue
        
        # 3. [NEW] 인물 필터링 (핵심!)
        if filter_person and filter_person != "All":
            # 이 일기에 등장한 사람들 이름 목록 만들기
            diary_people_names = [p['name'] for p in diary.get('people', [])]
            # 찾는 사람이 없으면 탈락
            if filter_person not in diary_people_names:
                continue
            
        filtered_diaries.append(diary)

    return render_template('history.html', 
                           diaries=filtered_diaries,
                           all_people=all_people_list, # 명단 전달
                           current_emotion=filter_emotion,
                           current_date=filter_date,
                           current_person=filter_person) # 현재 선택된 사람 전달

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        raw_date = request.form['date']
        diary_text = request.form['diary']
        english_date = format_english_date(raw_date)
        
        result = improved_analyzer.analyze_emotion_and_color(diary_text)
        
        diary_data = {
            'date': english_date,
            'text': diary_text,
            'emotion': result['emotion'],
            'color': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone'],
            'people': result['people']
        }
        save_diary(diary_data)
        
        return render_template('result.html', 
                               date=english_date,
                               text=diary_text,
                               emotion=result['emotion'],
                               color=result['color_hex'],
                               color_name=result['color_name'],
                               tone=result['tone'],
                               people=result['people'])

if __name__ == '__main__':
    app.run(debug=True)