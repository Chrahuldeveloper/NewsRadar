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


# ---------------- HELPERS ----------------
def resolve_city(city: str, state: str) -> str:
    """
    City should NEVER be empty.
    If city is blank, fall back to state.
    If both blank, use 'Global'.
    """
    city  = (city  or "").strip()
    state = (state or "").strip()
    if city:
        return city
    if state:
        return state
    return "Global"


def resolve_state(state: str) -> str:
    """State falls back to 'Global' when missing."""
    state = (state or "").strip()
    return state if state else "Global"


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
            model="deepseek-chat",
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
    Insert into content_radar_news_scrape.
    - city is NEVER empty: falls back to state, then 'Global'.
    - state falls back to 'Global'.
    - Uses (tittle, city, state) as the unique conflict key so the same
      headline from different locations each gets its own row.
    """
    tittle = (tittle or "").strip()
    if not tittle:
        return

    resolved_city  = resolve_city(city, state)
    resolved_state = resolve_state(state)

    try:
        # First try insert; if it already exists for this exact
        # (tittle, city, state) combo just update city/state in case
        # they changed (they won't, but keeps the upsert safe).
        supabase.table("content_radar_news_scrape").upsert(
            {
                "tittle": tittle,
                "city":   resolved_city,
                "state":  resolved_state,
            },
            on_conflict="tittle,city,state",   # composite key — add this UNIQUE constraint in Supabase
            ignore_duplicates=True             # skip silently if exact duplicate
        ).execute()

        print(f"  📝 Scrape saved [{resolved_city} / {resolved_state}]: {tittle[:60]}")

    except Exception as e:
        # Fallback: plain insert so we never silently lose data
        print(f"  ⚠️  Scrape upsert failed — trying plain insert [{resolved_city}]: {e}")
        try:
            supabase.table("content_radar_news_scrape").insert(
                {
                    "tittle": tittle,
                    "city":   resolved_city,
                    "state":  resolved_state,
                }
            ).execute()
        except Exception as e2:
            print(f"  ❌ Scrape insert also failed [{resolved_city}]: {e2}")


# ---------------- AI INTELLIGENCE ----------------
async def ai_itellengence(article: dict):
    """
    Runs AI on a single article dict and saves result to content_radar.
    City is NEVER empty (falls back to state / 'Global').
    Conflict key is (regular_tittle, city, state) so the same headline
    from different locations gets its own row.
    """
    print("🤖 AI running...")

    raw_tittle = (article.get("tittle") or "").strip()
    if not raw_tittle:
        return

    city  = resolve_city(article.get("city", ""), article.get("state", ""))
    state = resolve_state(article.get("state", ""))

    optimized = await optimise_tittle(raw_tittle)

    location_context = f"\nLocation Context: {city}, {state}"

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
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user", "content": prompt}
                ]
            ).choices[0].message.content.strip()

        output = await asyncio.to_thread(call_model)

        if output.strip().upper() == "SKIP":
            print(f"  ⏭️ Skipped [{city} / {state}]: {raw_tittle[:60]}")
            return

        # Upsert with composite conflict key.
        # ⚠️  Make sure Supabase has: UNIQUE (regular_tittle, city, state)
        supabase.table("content_radar").upsert(
            {
                "tittle":         optimized,
                "regular_tittle": raw_tittle,
                "summary":        output,
                "city":           city,    # NEVER empty
                "state":          state,   # NEVER empty
            },
            on_conflict="regular_tittle,city,state"
        ).execute()

        print(f"  ✅ Saved → content_radar [{city} / {state}]: {optimized}")

    except Exception as e:
        print(f"  ❌ AI error [{city} / {state}]: {e}")


# ---------------- SCRAPE BBC ----------------
async def scrape():
    """BBC headlines — saved as India / Global news."""
    print("\n📰 Scraping BBC...")
    try:
        r = requests.get(bbc, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        content = soup.find("div", class_="sc-cd6075cf-0 cJhFtM")
        if not content:
            print("  ⚠️  No BBC container found — BBC may have changed its HTML structure")
            return

        headlines = content.find_all("p")

        for h in headlines:
            text = h.get_text(strip=True)
            if not text or len(text) < 5:
                continue

            # BBC is international; city = "Global", state = "Global"
            article = {"tittle": text, "city": "", "state": ""}
            save_to_scrape_table(text, "", "")

            print(f"  BBC: {text[:80]}")
            await ai_itellengence(article)

    except Exception as e:
        print("  ❌ Scrape error:", e)


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
            title = (item.get("title") or "").strip()
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

        res2 = requests.get(
            "https://gnews.io/api/v4/search",
            params={"lang": "en", "country": "in", "max": 10, "apikey": gnews_key, "q": query},
            timeout=15
        ).json()
        for item in res2.get("articles", []):
            title = (item.get("title") or "").strip()
            if title:
                articles.append({"tittle": title, "city": city, "state": state})

    except Exception as e:
        print(f"  ❌ API error [{label}]:", e)
        return

    articles = articles[:max_results]

    for article in articles:
        save_to_scrape_table(article["tittle"], city, state)
        print(f"  Saved [{label}]:", article["tittle"][:70])
        await ai_itellengence(article)


# ---------------- FETCH INDIA CITY NEWS ----------------
async def fetch_india_city_news():
    print("\n🇮🇳 Fetching India city news (1 article per city)...")

    for loc in INDIA_LOCATIONS:
        city  = loc["city"]
        state = loc["state"]
        label = f"{city}, {state}"
        print(f"\n  📍 {label}")

        query   = f"{city} {state}".strip()
        article = None

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = (item.get("title") or "").strip()
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
                    title = (item.get("title") or "").strip()
                    if title:
                        article = {"tittle": title, "city": city, "state": state}
                        break

        except Exception as e:
            print(f"  ❌ API error [{label}]:", e)

        if article:
            # city and state are always populated here from INDIA_LOCATIONS
            save_to_scrape_table(article["tittle"], city, state)
            print(f"  ✅ Saved [{city}]:", article["tittle"][:70])
            await ai_itellengence(article)
        else:
            print(f"  ⚠️  No article found for {label}")

        await asyncio.sleep(1)


# ---------------- FETCH INDIA STATE NEWS ----------------
async def fetch_india_state_news():
    print("\n🗺️ Fetching India state news (5 per state)...")

    for loc in INDIA_STATES:
        state = loc["state"]
        # city is intentionally blank for state-level news;
        # resolve_city() will substitute state so DB city is never empty.
        city  = ""
        label = state
        print(f"\n  📍 {label}")

        query    = state
        articles = []

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "country": "in", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = (item.get("title") or "").strip()
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
                    title = (item.get("title") or "").strip()
                    if title:
                        articles.append({"tittle": title, "city": city, "state": state})
                    if len(articles) >= 5:
                        break

        except Exception as e:
            print(f"  ❌ API error [{label}]:", e)

        for article in articles:
            save_to_scrape_table(article["tittle"], city, state)
            print(f"  Saved [{state}]:", article["tittle"][:70])
            await ai_itellengence(article)

        await asyncio.sleep(1)


# ---------------- FETCH GLOBAL NEWS ----------------
async def fetch_global_news():
    print("\n🌍 Fetching global news (1 per query)...")

    for query in GLOBAL_QUERIES:
        print(f"\n  🌐 {query}")
        article_title = None

        try:
            res = requests.get(
                "https://newsdata.io/api/1/latest",
                params={"apikey": newsdata_api_key, "language": "en", "q": query},
                timeout=15
            ).json()
            for item in res.get("results", []):
                title = (item.get("title") or "").strip()
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
                    title = (item.get("title") or "").strip()
                    if title:
                        article_title = title
                        break

        except Exception as e:
            print(f"  ❌ API error [global/{query}]:", e)

        if article_title:
            # city="" state="" → resolve_city returns "Global", resolve_state returns "Global"
            save_to_scrape_table(article_title, "", "")
            print(f"  ✅ Saved [Global]:", article_title[:70])
            await ai_itellengence({"tittle": article_title, "city": "", "state": ""})
        else:
            print(f"  ⚠️  No article found for query: {query}")

        await asyncio.sleep(1)


# ---------------- FULL CYCLE ----------------
async def cycle():
    print("\n🧹 Clearing old data...")
    await clear_all_tables()

    await fetch_india_city_news()   # 8 cities, 1 each
    await fetch_india_state_news()  # 10 states, 5 each
    await fetch_global_news()       # 6 global queries, 1 each

    await scrape()                  # BBC

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