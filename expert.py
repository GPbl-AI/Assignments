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

# 요약 및 맥락 분석 (컨텍스트 기반 + CoT)
def summarize_and_analyze_with_context(articles, keyword, user_preferences):
    combined_text = " ".join([article["description"] for article in articles if article.get("description")])
    preferences = f"""
    Preferred categories: {', '.join(user_preferences['preferred_categories'])}
    Preferred style: {user_preferences['preferred_style']}
    Reading time: {user_preferences['reading_time']}
    """

    # 컨텍스트 기반 + CoT 프롬프트
    prompt = f"""
    You are a news analyst for a user with the following preferences:
    {preferences}

    Step 1: Summarize the following articles related to '{keyword}' in one or two paragraphs, focusing on recurring themes:
    {combined_text}

    Step 2: Based on the summary, analyze the context:
    - Historical context of the topic
    - Hidden causes or factors influencing the topic
    - Predicted future impacts
    - Generate three unanswered questions related to the topic.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional news analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"요약 및 맥락 분석 실패: {e}")
        return "요약 및 맥락 분석 실패."

# 분석 결과 저장
def save_analysis_results(keyword, analysis_result):
    filename = f"{keyword.replace(' ', '_').lower()}_analysis.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=4)
    print(f"Analysis results saved to {filename}")

# 메인 실행 흐름
if __name__ == "__main__":
    # 사용자 선호 데이터
    user_preferences = {
        "preferred_categories": ["AI", "health", "technology"],  # 선호 카테고리
        "preferred_style": "in-depth analysis",  # 기사 스타일
        "reading_time": "morning"  # 뉴스 소비 시간
    }

    keywords = user_preferences["preferred_categories"]
    for keyword in keywords:
        articles = fetch_news(keyword, count=50)
        relevant_articles = filter_articles_by_relevance(articles, keyword, threshold=0.3)

        if relevant_articles:
            analysis_result = summarize_and_analyze_with_context(relevant_articles, keyword, user_preferences)
            print(f"====== {keyword.upper()} Analysis ======")
            print(analysis_result)

            # 결과 저장
            save_analysis_results(keyword, {
                "keyword": keyword,
                "analysis": analysis_result,
                "articles": relevant_articles
            })
        else:
            print(f"No relevant articles found for keyword '{keyword}'.")