from bs4 import BeautifulSoup
from supabase import create_client
import os
import re
import requests
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# import numpy as np
# from sentence_transformers import SentenceTransformer
# from pytrends.request import TrendReq
# from google import genai
# import schedule
# import time


load_dotenv()
gnews_key = os.getenv("GnewsApi")
newsdata_api_key = os.getenv("Newsdata_api_key")

url= os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")


supabase = create_client(url, key)


bbc = "https://www.bbc.com/"




async def clear_news_table():
    try:
        supabase.table("news").delete().neq("id", 0).execute()
        print("🧹 News table cleared")
    except Exception as e:
        print("Cleanup failed:", e)


async def optimise_tittle(tittle: str):
    prompt = f"""
You are a NEWS TOPIC TAG generator.

TASK:
Convert the news into a short topic label.

RULES:
- Output ONLY 2 to 5 words
- NO sentences
- NO punctuation
- NO explanation
- NO opinions
- NO full phrases

STYLE EXAMPLES:
Attack on Donald Trump
Iran Israel Conflict
Strait of Hormuz Tension
US China Trade War
Middle East Crisis

INPUT:
{tittle}
"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "Return only 2-5 word topic label."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        output = response.choices[0].message.content.strip()

        return output

    except Exception as e:
        print("DeepSeek error:", e)
        return None    

    

async def ai_itellengence(article):
    print("ai ready (DeepSeek)")

    optimized_title = await optimise_tittle(article.get("tittle"))

    if not optimized_title:
        optimized_title = article.get("tittle")

    prompt = f"""
You are an expert Viral Content Strategist + Startup Idea Generator.

News Title:
{optimized_title}

STEP 0: VELOCITY SCORE (0-100)
- If <50 → respond ONLY "SKIP"

STEP 1: VIRAL CONTENT
1. Score
2. 3 Hooks
3. Emotion
4. Script
5. Hashtags
"""

    try:
        def call_model():
            response = deepseek_client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": "You generate viral content + startup ideas."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()

        output = await asyncio.to_thread(call_model)

        if output == "SKIP":
            return

        db_data = {
            "tittle": optimized_title,
            "regular_tittle": article.get("tittle"),
            "summary": output
        }

        supabase.table("content_radar").upsert(
            db_data,
            on_conflict="regular_tittle"
        ).execute()

        print("Saved with optimized title")
        return output

    except Exception as e:
        print("DeepSeek error:", e)
        return None
    


def clean_keywords(text):
    stop_words = {
    "the","is","a","an","in","on","at","to","and","but","says",
    "did","not","was","were","of","for","with","by","as","from","this","that",
    "it", "its", "they", "them", "their", "who", "which", "what", "where", "when",
    "how", "why", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "too", "very", "can", "will", "just", "should",
    "now", "about", "after", "before", "during", "under", "over", "between", 
    "into", "through", "breaking", "latest", "report", "update", "watch",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "up", "down", "out", "off", 
    "over", "under", "again", "further", "then", "once"
    "today", "yesterday", "tomorrow", "daily", "weekly", "month", "year", "years", 
    "time", "days", "hours", "minutes", "new", "old", "first", "last", "next", 
    "recent", "past",
    "report", "reports", "reported", "breaking", "update", "latest", "exclusive", 
    "official", "source", "sources", "according", "confirmed", "told", "claims", 
    "claimed", "details", "video", "watch", "live", "shared", "posted",
    "it", "its", "they", "them", "their", "who", "whom", "which", "what", "where", 
    "when", "how", "why", "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "too", "very", "can", "will", "just", 
    "should", "now"
       
   }

    words = re.findall(r"[a-zA-Z]+", text.lower())

    keywords = [w for w in words if w not in stop_words]

    return keywords[:5]



async def calculate_cos(article):
    try:
        await ai_itellengence(article=article)
        print("ai answer saved")

        await asyncio.sleep(2)

    except Exception as e:
        print("Processing error:", e)

        await asyncio.sleep(5)
        try:
            await ai_itellengence(article=article)
        except Exception as e:
            print("Retry failed:", e)


async def getting_and_scroing_articles(article):
    await calculate_cos(article=article)


# def cosine_similarity(a,b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def scrape(url, section_container, inner_section, element, id):

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    if id:
        return

    content = soup.find(section_container, class_=inner_section)

    headlines = content.find_all(element)

    # res = supabase.table("news").select("Vectors").execute()

    # existing_vectors = []

    # for row in res.data:
    #     vec = row.get("Vectors")
    #     if vec:
    #         existing_vectors.append(vec)

    # print(f"Loaded {len(existing_vectors)} existing embeddings")

    for title in headlines:

        text = title.get_text(strip=True)

        if not text:
            continue

        # embedding = get_model().encode(text).tolist()

        is_duplicate = False

        # for vec in existing_vectors:
        #     try:
        #         score = cosine_similarity(embedding, vec)

        #         if score > 0.85:
        #             print("Duplicate skipped:", text)
        #             is_duplicate = True
        #             break
        #     except:
        #         continue

        # if is_duplicate:
        #     continue

        article = {
            "tittle": text,
            # "Vectors": embedding
        }

        try:
            supabase.table("news").upsert(article, on_conflict="tittle").execute()
            print("Saved:", text)
            # existing_vectors.append(embedding)
            await getting_and_scroing_articles(article)
        except Exception as e:
            print("Failed:", text)
            print(e)

        print("-" * 80)



async def get_data_via_api():

    articles = []
    res = requests.get(f"https://newsdata.io/api/1/latest?apikey={newsdata_api_key}")
    data = res.json()

    for item in data.get("results", []):

        if not isinstance(item, dict):
            continue

        title = item.get("regular_tittle")   

        if not title:
            continue

        # embedding = get_model().encode(title).tolist()

        articles.append({
            "regular_tittle": title,
            # "Vectors": embedding
        })

    res2 =  requests.get(
        f"https://gnews.io/api/v4/search?q=example&lang=en&country=in&max=10&apikey={gnews_key}"
    )
    data2 = res2.json()

    for item in data2.get("articles", []):

        if not isinstance(item, dict):
            continue

        title = item.get("regular_tittle")  

        if not title:
            continue

        # embedding = get_model().encode(title).tolist()

        articles.append({
            "regular_tittle": title,
            # "Vectors": embedding   
        })
    # res_db = supabase.table("news").select("Vectors").execute()

    # existing_vectors = [
    #     row["Vectors"]
    #     for row in res_db.data
    #     if row.get("Vectors")
    # ]

    # THRESHOLD = 0.85

    for article in articles:

        try:
            # embedding = article["Vectors"]

            is_duplicate = False

            # for vec in existing_vectors:
            #     try:
            #         score = cosine_similarity(embedding, vec)

            #         if score > THRESHOLD:
            #             print("Duplicate skipped:", article["tittle"])
            #             is_duplicate = True
            #             break
            #     except:
            #         continue

            # if is_duplicate:
            #     continue

            supabase.table("news").upsert(article, on_conflict="tittle").execute()

            print("Saved:", article["tittle"])

            # existing_vectors.append(embedding)
            await getting_and_scroing_articles(article)
            
        except Exception as e:
            print("Failed:", article["regular_tittle"])
            print(e)



# bbc = "https://www.bbc.com/"

async def cycle():
    print("🧹 Clearing old data...")
    await clear_news_table()

    print("🚀 Running API fetch...")
    await get_data_via_api()

    print("🧹 Running scrape...")
    await scrape(
        bbc,
        section_container='div',
        inner_section='sc-cd6075cf-0 cJhFtM',
        element='p',
        id=False
    )

async def main():
    while True:
        try:
            await cycle()
            print("✅ Cycle complete")

        except Exception as e:
            print("❌ Cycle error:", e)

        await asyncio.sleep(12 * 60 * 60)

if __name__ == "__main__":
    asyncio.run(main())