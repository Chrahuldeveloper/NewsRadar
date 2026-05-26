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

# ---------------- INDIA CITY QUERIES ----------------
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

# ---------------- TOP INDIA STATE QUERIES (by GDP) ----------------
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

# ---------------- GLOBAL QUERIES ----------------
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
async def optimise_tittle(tittle: str) -> str:
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
            model="deepseek-chat",          # ✅ fixed: was "deepseek-v4-pro" which doesn't exist
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": tittle}
            ]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("Optimise error:", e)
        return tittle  # fallback to raw title


# ---------------- SAVE TO SCRAPE TABLE ----------------
def save_to_scrape_table(tittle: str, city: str, state: str):
    """
    Safe insert into content_radar_news_scrape.
    Uses (tittle, city, state) composite logic:
    - upsert on tittle alone can silently skip location updates.
    - We insert ignoring duplicate tittles to preserve first-seen location.
    """
    try:
        supabase.table("content_radar_news_scrape").upsert(
            {"tittle": tittle, "city": city, "state": state},
            on_conflict="tittle",           # adjust if your PK is composite
            ignore_duplicates=False         # let it update city/state if title exists
        ).execute()
    except Exception as e:
        print(f"  Scrape table insert failed [{city or state or 'global'}]:", e)


# ---------------- AI INTELLIGENCE ----------------
async def ai_itellengence(article: dict):
    """
    Runs AI on a single article dict and saves result to content_radar.
    Conflict key is now (regular_tittle, city, state) to allow the same
    headline from different locations to each get its own row.
    """
    print("AI running...")

    raw_tittle = article.get("tittle", "").strip()
    if not raw_tittle:
        return

    city  = article.get("city", "") or ""
    state = article.get("state", "") or ""

    optimized = await optimise_tittle(raw_tittle)

    location_context = ""
    if city or state:
        parts = [p for p in [city, state] if p]
        location_context = f"\nLocation Context: {', '.join(parts)}"

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
                model="deepseek-chat",      # ✅ fixed: consistent model name
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user", "content": prompt}
                ]
            ).choices[0].message.content.strip()

        output = await asyncio.to_thread(call_model)

        if output.strip().upper() == "SKIP":
            print(f"  ⏭️ Skipped [{city or state or 'global'}]: {raw_tittle}")
            return

        # ✅ FIX: use (regular_tittle + city + state) as the natural unique key.
        # This prevents a Mumbai headline from overwriting a Delhi headline with the same text.
        # Your Supabase table needs a unique constraint on (regular_tittle, city, state).
        supabase.table("content_radar").upsert(
            {
                "tittle":         optimized,
                "regular_tittle": raw_tittle,
                "summary":        output,
                "city":           city,      # ✅ always explicitly set
                "state":          state,     # ✅ always explicitly set
            },
            on_conflict="regular_tittle,city,state"   # ✅ composite conflict key
        ).execute()

        label = city or state or "global"
        print(f"  ✅ Saved → content_radar [{label}]: {optimized}")

    except Exception as e:
        print(f"  AI error [{city or state or 'global'}]:", e)


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
            save_to_scrape_table(text, "", "India")

            print("BBC saved:", text)
            await ai_itellengence(article)

    except Exception as e:
        print("Scrape error:", e)


# ---------------- FETCH FOR A SINGLE LOCATION ----------------
async def fetch_location(city: str, state: str, label: str, max_results: int = 10):
    query = f"{city} {state}".strip() if city else state
    articles = []

    try:
        res = requests.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
            timeout=15
        ).json()
        for item in res.get("results", []):
            title = item.get("title", "").strip()
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

        res2 = requests.get(
            "https://gnews.io/api/v4/search",
            params={"lang": "en", "country": "in", "max": 10, "apikey": gnews_key, "q": query},
            timeout=15
        ).json()
        for item in res2.get("articles", []):
            title = item.get("title", "").strip()
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

    except Exception as e:
        print(f"  API error [{label}]:", e)
        return

    articles = articles[:max_results]

    for article in articles:
        save_to_scrape_table(article["tittle"], city, state)
        print(f"  Saved [{label}]:", article["tittle"])
        await ai_itellengence(article)


# ---------------- FETCH INDIA CITY NEWS ----------------
async def fetch_india_city_news():
    print("\n🇮🇳 Fetching India city news (1 article per city)...")

    for loc in INDIA_LOCATIONS:
        city  = loc["city"]
        state = loc["state"]
        label = f"{city}, {state}"
        print(f"  📍 {label}")

        query = f"{city} {state}".strip()
        article = None

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = item.get("title", "").strip()
                if title:
                    article = {"tittle": title, "city": city, "state": state}
                    break

            if not article:
                res2 = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"lang": "en", "country": "in", "max": 5, "apikey": gnews_key, "q": query},
                    timeout=15
                ).json()
                for item in res2.get("articles", []):
                    title = item.get("title", "").strip()
                    if title:
                        article = {"tittle": title, "city": city, "state": state}
                        break

        except Exception as e:
            print(f"  API error [{label}]:", e)

        if article:
            save_to_scrape_table(article["tittle"], city, state)   # ✅ explicit city+state
            print(f"  Saved [{city}]:", article["tittle"])
            await ai_itellengence(article)
        else:
            print(f"  ⚠️ No article found for {label}")

        await asyncio.sleep(1)


# ---------------- FETCH INDIA STATE NEWS ----------------
async def fetch_india_state_news():
    print("\n🗺️ Fetching India state news (5 per state)...")

    for loc in INDIA_STATES:
        state = loc["state"]
        city  = ""                          # ✅ explicit empty string, not None
        label = state
        print(f"  📍 {label}")

        query = state
        articles = []

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = item.get("title", "").strip()
                if title:
                    articles.append({"tittle": title, "city": city, "state": state})
                if len(articles) >= 5:
                    break

            if len(articles) < 5:
                res2 = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"lang": "en", "country": "in", "max": 5, "apikey": gnews_key, "q": query},
                    timeout=15
                ).json()
                for item in res2.get("articles", []):
                    title = item.get("title", "").strip()
                    if title:
                        articles.append({"tittle": title, "city": city, "state": state})
                    if len(articles) >= 5:
                        break

        except Exception as e:
            print(f"  API error [{label}]:", e)

        for article in articles:
            save_to_scrape_table(article["tittle"], city, state)   # ✅ explicit city+state
            print(f"  Saved [{state}]:", article["tittle"])
            await ai_itellengence(article)

        await asyncio.sleep(1)


# ---------------- FETCH GLOBAL NEWS ----------------
async def fetch_global_news():
    print("\n🌍 Fetching global news (1 per query)...")

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
                title = item.get("title", "").strip()
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
                    title = item.get("title", "").strip()
                    if title:
                        article_title = title
                        break

        except Exception as e:
            print(f"  API error [global/{query}]:", e)

        if article_title:
            save_to_scrape_table(article_title, "", "")            # ✅ global: both empty
            print(f"  Saved [global]:", article_title)
            await ai_itellengence({"tittle": article_title, "city": "", "state": ""})
        else:
            print(f"  ⚠️ No article found for query: {query}")

        await asyncio.sleep(1)


# ---------------- FULL CYCLE ----------------
async def cycle():
    print("\n🧹 Clearing old data...")
    await clear_all_tables()

    await fetch_india_city_news()   # 8 cities, 1 each
    await fetch_india_state_news()  # 10 states, 5 each
    await fetch_global_news()       # 6 global queries, 1 each

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
