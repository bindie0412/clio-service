from flask import Flask, render_template, request
import json
import os
import random
import requests
import base64
from datetime import datetime
from our_model.emotion_model import improved_analyzer

app = Flask(__name__)

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, 'diaries.json')

# --- Spotify 설정 ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

EMOTION_QUERIES = {
    "joy": ["happy mood", "feel-good", "upbeat pop", "party"],
    "sadness": ["sad", "ballad", "emotional", "piano"],
    "anger": ["angry", "rock", "metal", "hard"],
    "fear": ["calm", "relaxing", "comfort"],
    "surprise": ["exciting", "electronic", "EDM"],
    "disgust": ["chill", "lo-fi", "indie"],
}

def get_spotify_token():
    """Client Credentials Flow 로 access token 발급"""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "client_credentials"}
    resp = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    if resp.status_code != 200:
        print("⚠️ Spotify token error:", resp.text)
        return None
    return resp.json().get("access_token")

def get_random_track_by_emotion(emotion: str):
    """감정에 맞는 검색어로 Spotify에서 랜덤 트랙 하나 가져오기"""
    token = get_spotify_token()
    if not token:
        return None

    query_list = EMOTION_QUERIES.get(emotion, ["mood"])
    query = random.choice(query_list)

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "type": "track",
        "limit": 20,
        "market": "KR",
    }
    resp = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    if resp.status_code != 200:
        print("⚠️ Spotify search error:", resp.text)
        return None

    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None

    track = random.choice(items)
    return track.get("external_urls", {}).get("spotify")

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

# --- 모든 등장인물 이름 모으기 ---
def get_all_people():
    diaries = load_diaries()
    people_set = set()
    for diary in diaries:
        if 'people' in diary:
            for p in diary['people']:
                people_set.add(p['name'])
    return sorted(list(people_set))

# ================= 라우팅 =================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    filter_emotion = request.args.get('emotion')
    filter_date = request.args.get('date')
    filter_person = request.args.get('person')

    all_diaries = load_diaries()
    all_people_list = get_all_people()

    filtered_diaries = []
    
    for diary in all_diaries:
        if filter_emotion and filter_emotion != "All" and diary['emotion'] != filter_emotion:
            continue
        
        if filter_date:
            english_filter_date = format_english_date(filter_date)
            if diary['date'] != english_filter_date:
                continue
        
        if filter_person and filter_person != "All":
            diary_people_names = [p['name'] for p in diary.get('people', [])]
            if filter_person not in diary_people_names:
                continue
            
        filtered_diaries.append(diary)

    return render_template(
        'history.html',
        diaries=filtered_diaries,
        all_people=all_people_list,
        current_emotion=filter_emotion,
        current_date=filter_date,
        current_person=filter_person
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        raw_date = request.form['date']
        diary_text = request.form['diary']
        english_date = format_english_date(raw_date)
        
        result = improved_analyzer.analyze_emotion_and_color(diary_text)

        # Spotify에서 감정에 맞는 랜덤 트랙 하나 가져오기
        spotify_link = get_random_track_by_emotion(result['emotion'])
        
        diary_data = {
            'date': english_date,
            'text': diary_text,
            'emotion': result['emotion'],
            'color': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone'],
            'people': result['people'],
        }
        save_diary(diary_data)
        
        return render_template(
            'result.html',
            date=english_date,
            text=diary_text,
            emotion=result['emotion'],
            color=result['color_hex'],
            color_name=result['color_name'],
            tone=result['tone'],
            people=result['people'],
            spotify_link=spotify_link,
        )

if __name__ == '__main__':
    app.run(debug=True)
