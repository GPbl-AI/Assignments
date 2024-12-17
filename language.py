from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import feedparser
import requests
import openai
import logging
import os
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# BERT 모델 초기화
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 임베딩 생성 함수
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# 뉴스 가져오기 (RSS + NewsAPI 통합)
def fetch_news(topic, count=10):
    articles = []
    # RSS 피드 가져오기
    try:
        encoded_topic = topic.replace(" ", "%20")
        rss_url = f"https://news.google.com/rss/search?q={encoded_topic}"
        feed = feedparser.parse(rss_url)
        articles += [
            {
                "title": entry.title,
                "link": entry.link,
                "description": BeautifulSoup(entry.summary, "html.parser").get_text(),
                "publishedAt": datetime(*entry.published_parsed[:6]) if entry.published_parsed else None
            }
            for entry in feed.entries[:count]
        ]
    except Exception as e:
        logging.warning(f"RSS 피드 가져오기 실패: {e}")

    # NewsAPI 가져오기
    if NEWSAPI_KEY:
        try:
            api_url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={count}&apiKey={NEWSAPI_KEY}"
            response = requests.get(api_url)
            response.raise_for_status()
            articles += [
                {
                    "title": article["title"],
                    "link": article["url"],
                    "description": article["description"],
                    "publishedAt": datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                }
                for article in response.json().get("articles", [])
            ]
        except Exception as e:
            logging.warning(f"NewsAPI 호출 실패: {e}")

    return articles

# 기사 필터링
def filter_articles_by_relevance(articles, keyword, threshold=0.5):
    keyword_embedding = get_embedding(keyword)
    relevant_articles = []
    for article in articles:
        combined_text = article["title"] + " " + (article.get("description") or "")
        article_embedding = get_embedding(combined_text)
        similarity = cosine_similarity(keyword_embedding, article_embedding)[0][0]
        if similarity >= threshold:
            relevant_articles.append(article)
    return relevant_articles

# 요약 및 맥락 분석
def summarize_and_analyze_with_context(articles, keyword, user_preferences):
    # 기사를 하나로 병합
    combined_text = " ".join([article["description"] for article in articles if article.get("description")])
    preferences = f"""
    사용자의 뉴스 선호도는 다음과 같습니다:
    - 선호 카테고리: {', '.join(user_preferences['preferred_categories'])}
    - 선호 스타일: {user_preferences['preferred_style']}
    - 뉴스 읽는 시간: {user_preferences['reading_time']}
    """

    # 자연스러운 말투 강조
    prompt = f"""
    아래는 사용자의 선호도를 반영한 뉴스 요약 및 분석 요청입니다:
    {preferences}

    1단계: '{keyword}'와 관련된 기사를 한두 개의 단락으로 요약해주세요. 
    - 반복되는 주제와 핵심 내용을 중심으로 작성해주세요.
    - 간결하면서도 자연스럽고 쉽게 읽히는 문체로 작성해주세요.

    뉴스 기사 내용:
    {combined_text}

    2단계: 요약한 내용을 바탕으로 다음 질문에 답변해주세요:
    - 이 주제의 역사적 배경은 무엇인가요?
    - 이 주제와 관련된 숨겨진 원인이나 요인은 무엇인가요?
    - 이 주제가 미래에 미칠 영향은 무엇인가요?
    - 이 주제와 관련하여 답변되지 않은 중요한 질문 세 가지를 생성해주세요.
    """

    try:
        # GPT 모델 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 한국어 뉴스 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.7
        )
        # GPT 모델 응답 반환
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        # 오류 처리
        logging.error(f"요약 및 맥락 분석 실패: {e}")
        return "요약 및 맥락 분석 실패."

# 분석 결과 저장
def save_analysis_results(keyword, analysis_result):
    filename = f"{keyword.replace(' ', '_').lower()}_analysis_kr.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=4)
    print(f"분석 결과가 {filename}에 저장되었습니다.")

# 메인 실행 흐름
if __name__ == "__main__":
    # 사용자 선호 데이터
    user_preferences = {
        "preferred_categories": ["AI", "health", "technology"],  # 선호 카테고리
        "preferred_style": "심층 분석",  # 기사 스타일
        "reading_time": "아침"  # 뉴스 소비 시간
    }

    keywords = user_preferences["preferred_categories"]
    for keyword in keywords:
        articles = fetch_news(keyword, count=50)
        relevant_articles = filter_articles_by_relevance(articles, keyword, threshold=0.3)

        if relevant_articles:
            analysis_result = summarize_and_analyze_with_context(relevant_articles, keyword, user_preferences)
            print(f"====== {keyword.upper()} 분석 결과 ======")
            print(analysis_result)

            # 결과 저장
            save_analysis_results(keyword, {
                "keyword": keyword,
                "analysis": analysis_result,
                "articles": relevant_articles
            })
        else:
            print(f"'{keyword}'에 대한 관련 기사를 찾을 수 없습니다.")