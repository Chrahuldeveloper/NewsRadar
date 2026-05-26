from bs4 import BeautifulSoup
from supabase import create_client
import os
import requests
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

gnews_key = os.getenv("GnewsApi")
newsdata_api_key = os.getenv("Newsdata_api_key")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

bbc = "https://www.bbc.com/"

# ---------------- 10 INDIA CITY QUERIES ----------------
INDIA_LOCATIONS = [
    {"state": "Delhi",          "city": "New Delhi"},
    {"state": "Maharashtra",    "city": "Mumbai"},
    {"state": "Karnataka",      "city": "Bangalore"},
    {"state": "Tamil Nadu",     "city": "Chennai"},
    {"state": "West Bengal",    "city": "Kolkata"},
    {"state": "Telangana",      "city": "Hyderabad"},
    {"state": "Gujarat",        "city": "Ahmedabad"},
    {"state": "Maharashtra",    "city": "Pune"},
]

# ---------------- TOP 10 INDIA STATE QUERIES (by GDP) ----------------
INDIA_STATES = [
    {"state": "Maharashtra",    "city": ""},
    {"state": "Tamil Nadu",     "city": ""},
    {"state": "Karnataka",      "city": ""},
    {"state": "Gujarat",        "city": ""},
    {"state": "Uttar Pradesh",  "city": ""},
    {"state": "West Bengal",    "city": ""},
    {"state": "Rajasthan",      "city": ""},
    {"state": "Telangana",      "city": ""},
    {"state": "Andhra Pradesh", "city": ""},
    {"state": "Kerala",         "city": ""},
]

# ---------------- 6 GLOBAL QUERIES ----------------
GLOBAL_QUERIES = [
    "news today",
    "breaking news",
    "latest news",
    "world news",
    "international news",
    "global news",
]

# ---------------- CLEAR TABLES ----------------
async def clear_all_tables():
    try:
        supabase.table("content_radar_news_scrape").delete().neq("tittle", "").execute()
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

    city  = article.get("city", "")
    state = article.get("state", "")

    location_context = ""
    if city or state:
        location_context = f"\nLocation Context: {city}, {state}".strip(", ")

    prompt = f"""
You are a Viral Content Strategist.

News:
{optimized}{location_context}

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
            "tittle":         optimized,
            "regular_tittle": raw_tittle,
            "summary":        output,
            "city":           city,
            "state":          state
        }, on_conflict="regular_tittle").execute()

        print(f"Saved → content_radar [{city or state or 'global'}]")

    except Exception as e:
        print("AI error:", e)


# ---------------- SCRAPE BBC ----------------
async def scrape():
    """BBC headlines — saved as India national news."""
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

            article = {"tittle": text, "city": "", "state": "India"}

            supabase.table("content_radar_news_scrape").upsert(
                {"tittle": text, "city": "", "state": "India"},
                on_conflict="tittle"
            ).execute()

            print("BBC saved:", text)
            await ai_itellengence(article)

    except Exception as e:
        print("Scrape error:", e)


# ---------------- FETCH FOR A SINGLE LOCATION ----------------
async def fetch_location(city: str, state: str, label: str, max_results: int = 10):  # ✅ default 10
    query = f"{city} {state}".strip() if city else state
    articles = []

    try:
        res = requests.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
            timeout=15
        ).json()
        for item in res.get("results", []):
            title = item.get("title")
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

        res2 = requests.get(
            "https://gnews.io/api/v4/search",
            params={"lang": "en", "country": "in", "max": 10, "apikey": gnews_key, "q": query},
            timeout=15
        ).json()
        for item in res2.get("articles", []):
            title = item.get("title")
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

    except Exception as e:
        print(f"  API error [{label}]:", e)
        return

    articles = articles[:max_results]  # ✅ always slice

    for article in articles:
        try:
            supabase.table("content_radar_news_scrape").upsert(
                {"tittle": article["tittle"], "city": city, "state": state},
                on_conflict="tittle"
            ).execute()
            print(f"  Saved [{label}]:", article["tittle"])
            await ai_itellengence(article)
        except Exception as e:
            print("  Insert failed:", e)


# ---------------- FETCH INDIA CITY NEWS (all 10 cities, 1 each = 10 total) ----------------
async def fetch_india_city_news():
    print("\n🇮🇳 Fetching India city news (all cities, 1 each = 10 total)...")

    for loc in INDIA_LOCATIONS:
        label = f"{loc['city']}, {loc['state']}"
        print(f"  📍 {label}")
        query = f"{loc['city']} {loc['state']}".strip()
        article = None

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = item.get("title")
                if title:
                    article = {"tittle": title, "city": loc["city"], "state": loc["state"]}
                    break  # ✅ take only 1

            if not article:
                res2 = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"lang": "en", "country": "in", "max": 5, "apikey": gnews_key, "q": query},
                    timeout=15
                ).json()
                for item in res2.get("articles", []):
                    title = item.get("title")
                    if title:
                        article = {"tittle": title, "city": loc["city"], "state": loc["state"]}
                        break  # ✅ take only 1

        except Exception as e:
            print(f"  API error [{label}]:", e)

        if article:
            try:
                supabase.table("content_radar_news_scrape").upsert(
                    {"tittle": article["tittle"], "city": article["city"], "state": article["state"]},
                    on_conflict="tittle"
                ).execute()
                print(f"  Saved [{article['city']}]:", article["tittle"])
                await ai_itellengence(article)
            except Exception as e:
                print("  Insert failed:", e)

        await asyncio.sleep(1)



# ---------------- FETCH INDIA STATE NEWS (all 28 states, 5 each) ----------------
async def fetch_india_state_news():
    print("\n🗺️ Fetching India state news (28 states, 5 each)...")

    for loc in INDIA_STATES:
        label = loc["state"]
        print(f"  📍 {label}")
        query = loc["state"]
        articles = []

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = item.get("title")
                if title:
                    articles.append({"tittle": title, "city": "", "state": loc["state"]})
                if len(articles) >= 5:
                    break

            if len(articles) < 5:
                res2 = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"lang": "en", "country": "in", "max": 5, "apikey": gnews_key, "q": query},
                    timeout=15
                ).json()
                for item in res2.get("articles", []):
                    title = item.get("title")
                    if title:
                        articles.append({"tittle": title, "city": "", "state": loc["state"]})
                    if len(articles) >= 5:
                        break

        except Exception as e:
            print(f"  API error [{label}]:", e)

        for article in articles:
            try:
                supabase.table("content_radar_news_scrape").upsert(
                    {"tittle": article["tittle"], "city": "", "state": article["state"]},
                    on_conflict="tittle"
                ).execute()
                print(f"  Saved [{article['state']}]:", article["tittle"])
                await ai_itellengence(article)
            except Exception as e:
                print("  Insert failed:", e)

        await asyncio.sleep(1)


async def fetch_global_news():
    print("\n🌍 Fetching global news (6 queries, 1 each)...")

    for query in GLOBAL_QUERIES:
        print(f"  🌐 {query}")
        article_title = None

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = item.get("title")
                if title:
                    article_title = title
                    break

            if not article_title:
                res2 = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"lang": "en", "max": 5, "apikey": gnews_key, "q": query},
                    timeout=15
                ).json()
                for item in res2.get("articles", []):
                    title = item.get("title")
                    if title:
                        article_title = title
                        break

        except Exception as e:
            print(f"  API error [global/{query}]:", e)

        if article_title:
            try:
                supabase.table("content_radar_news_scrape").upsert(
                    {"tittle": article_title, "city": "", "state": ""},
                    on_conflict="tittle"
                ).execute()
                print(f"  Saved [global]:", article_title)
                await ai_itellengence({"tittle": article_title, "city": "", "state": ""})
            except Exception as e:
                print("  Insert failed:", e)

        await asyncio.sleep(1)

# ---------------- FULL CYCLE ----------------
async def cycle():
    print("\n🧹 Clearing old data...")
    await clear_all_tables()

    await fetch_india_city_news()   # 10 cities
    await fetch_india_state_news()  # 6 states
    await fetch_global_news()       # 6 global topics

    print("\n📰 Scraping BBC...")
    await scrape()

    print("\n✅ Full cycle complete — next run in 24 hours")


# ---------------- MAIN LOOP ----------------
async def main():
    while True:
        try:
            await cycle()
        except Exception as e:
            print("❌ Cycle error:", e)

        await asyncio.sleep(24 * 60 * 60)  # 24 hours


if __name__ == "__main__":
    asyncio.run(main())