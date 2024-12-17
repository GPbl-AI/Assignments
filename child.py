from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import feedparser
import requests
import openai
import logging
import os
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

# 뉴스 수집 함수
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

# 기사 필터링 함수
def filter_articles_by_relevance(articles, keyword, threshold=0.3):
    keyword_embedding = get_embedding(keyword)
    relevant_articles = []
    for article in articles:
        combined_text = article["title"] + " " + (article.get("description") or "")
        article_embedding = get_embedding(combined_text)
        similarity = cosine_similarity(keyword_embedding, article_embedding)[0][0]
        if similarity >= threshold:
            relevant_articles.append(article)
    return relevant_articles

# 기사 요약 생성
def summarize_articles(articles):
    if not articles:
        return "요약할 내용이 없습니다."
    combined_text = " ".join([article["description"] for article in articles if article.get("description")])
    try:
        prompt = f"""
        Summarize the following news articles in one or two paragraphs, focusing on recurring themes:
        {combined_text}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"요약 생성 실패: {e}")
        return "요약 생성 실패."

# 어린이용 쉬운 표현 생성
def simplify_for_kids(summary, use_cot=False):
    if use_cot:
        prompt = f"""
        You are a teacher explaining complex topics to a child. Break down the explanation step by step to make it easy and fun:
        
        Original: "{summary}"
        
        Step 1: 
        Step 2: 
        Final: 
        """
    else:
        prompt = f"""
        You are an expert at explaining complex ideas to children in a simple and fun way. Here are some examples:

        1. Original: "Artificial intelligence is a field of computer science that simulates human intelligence."
           Simplified: "AI is like teaching computers to think and solve problems like humans do."

        2. Original: "Photosynthesis is the process by which plants use sunlight to make food."
           Simplified: "Plants eat sunlight and turn it into food. It's like magic for plants!"

        Now, rewrite the following text so that a 7-year-old child can easily understand it:
        {summary}
        """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"어린이용 변환 실패: {e}")
        return "어린이용 버전 생성 실패."

# 이미지 생성 함수
def generate_image(prompt):
    try:
        # 프롬프트 길이 제한 (700자 이내로 축소)
        if len(prompt) > 800:
            prompt = prompt[:797] + "..."
        
        # "귀여운 어린이용 이미지" 추가
        prompt = f"Create a cute and colorful cartoon-style image for kids: {prompt}"
        
        response = openai.Image.create(
            prompt=prompt,
            n=1,  # 생성할 이미지 개수
            size="512x512"  # 이미지 크기
        )
        image_url = response['data'][0]['url']
        return image_url
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI 이미지 생성 실패: {e}")
        return "이미지 생성 실패."
    except Exception as e:
        logging.error(f"알 수 없는 오류: {e}")
        return "이미지 생성 실패."

# 메인 실행 흐름
if __name__ == "__main__":
    keywords = ["korea"]
    for keyword in keywords:
        articles = fetch_news(keyword, count=50)
        relevant_articles = filter_articles_by_relevance(articles, keyword, threshold=0.3)
        if relevant_articles:
            # 기사 요약
            summary = summarize_articles(relevant_articles)
            
            # 어린이용 쉬운 표현
            simplified_summary = simplify_for_kids(summary, use_cot=False)
            
            # 어린이용 이미지 생성
            image_prompt = f"Create a fun and colorful image for kids about: {simplified_summary}"
            image_url = generate_image(image_prompt)
            
            # 출력
            print(f"====== {keyword.upper()} Summary ======")
            print(summary)
            print("\n====== Kid-Friendly Version ======")
            print(simplified_summary)
            print("\n====== Image URL ======")
            print(image_url)
            print("\n====== Relevant Articles ======")
            # for article in relevant_articles:
            #     print(f"Title: {article['title']}")
            #     print(f"Link: {article['link']}")
            #     print(f"Published At: {article['publishedAt']}\n")
        else:
            print(f"No relevant articles found for keyword '{keyword}'.")