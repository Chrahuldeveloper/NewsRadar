from bs4 import BeautifulSoup
from supabase import create_client
import os
import re
import requests
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

gnews_key = os.getenv("GnewsApi")
newsdata_api_key = os.getenv("Newsdata_api_key")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

supabase = create_client(url, key)

bbc = "https://www.bbc.com/"


# ---------------- CLEAR TABLES ----------------
async def clear_all_tables():
    try:
        supabase.table("news").delete().neq("tittle", "").execute()
        supabase.table("content_radar").delete().neq("tittle", "").execute()
        print("🧹 Tables cleared")
    except Exception as e:
        print("Cleanup error:", e)


# ---------------- OPTIMISE TITLE ----------------
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
        res = deepseek_client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": tittle}
            ]
        )
        return res.choices[0].message.content.strip()

    except Exception as e:
        print("Optimise error:", e)
        return tittle


# ---------------- AI INTELLIGENCE ----------------
async def ai_itellengence(article):
    print("AI running...")

    raw_tittle = article.get("tittle")
    if not raw_tittle:
        return

    optimized = await optimise_tittle(raw_tittle)

    prompt = f"""
You are a Viral Content Strategist.

News:
{optimized}

STEP 0: If low value → return SKIP

STEP 1:
1. Score
2. Hooks
3. Emotion
4. Script
5. Hashtags
"""

    try:
        def call_model():
            return deepseek_client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user", "content": prompt}
                ]
            ).choices[0].message.content.strip()

        output = await asyncio.to_thread(call_model)

        if output == "SKIP":
            return

        supabase.table("content_radar").upsert({
            "tittle": optimized,
            "regular_tittle": raw_tittle,
            "summary": output
        }, on_conflict="regular_tittle").execute()

        print("Saved → content_radar")

    except Exception as e:
        print("AI error:", e)


# ---------------- SCRAPE BBC ----------------
async def scrape():
    try:
        r = requests.get(bbc, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        content = soup.find("div", class_="sc-cd6075cf-0 cJhFtM")
        if not content:
            print("No BBC container found")
            return

        headlines = content.find_all("p")

        for h in headlines:
            text = h.get_text(strip=True)

            if not text or len(text) < 5:
                continue

            article = {"tittle": text}

            supabase.table("news").upsert(
                article,
                on_conflict="tittle"
            ).execute()

            print("Saved:", text)

            await ai_itellengence(article)

    except Exception as e:
        print("Scrape error:", e)


# ---------------- API FETCH ----------------
async def get_data_via_api():
    articles = []

    try:
        # -------- NewsData --------
        res = requests.get(
            f"https://newsdata.io/api/1/latest?apikey={newsdata_api_key}"
        ).json()

        for item in res.get("results", []):
            title = item.get("title")
            if title:
                articles.append({"tittle": title})

        # -------- GNews --------
        res2 = requests.get(
            f"https://gnews.io/api/v4/search?q=india&lang=en&country=in&max=10&apikey={gnews_key}"
        ).json()

        for item in res2.get("articles", []):
            title = item.get("title")
            if title:
                articles.append({"tittle": title})

    except Exception as e:
        print("API error:", e)
        return

    for article in articles:
        try:
            supabase.table("news").upsert(
                article,
                on_conflict="tittle"
            ).execute()

            print("Saved:", article["tittle"])

            await ai_itellengence(article)

        except Exception as e:
            print("Insert failed:", e)


# ---------------- CYCLE ----------------
async def cycle():
    print("🧹 Clearing old data...")
    await clear_all_tables()

    print("🚀 Fetching API...")
    await get_data_via_api()

    print("🧹 Scraping BBC...")
    await scrape()


# ---------------- MAIN LOOP ----------------
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