from bs4 import BeautifulSoup
from supabase import create_client
import os
import requests
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateutil_parser   

load_dotenv()

gnews_key         = os.getenv("GnewsApi")
newsdata_api_key  = os.getenv("Newsdata_api_key")

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

bbc = "https://www.bbc.com/"

_seen_titles: set[str] = set()

# ─── Freshness window ────────────────────────────────────────────────────────
MAX_AGE_HOURS = 48   # articles older than this are silently dropped

def _cutoff_dt() -> datetime:
    """UTC datetime representing the oldest article we'll accept."""
    return datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)

def _is_fresh(pub_date_str: str | None) -> bool:
    """
    Returns True when pub_date_str is within MAX_AGE_HOURS.
    If the date string is missing or unparseable, we REJECT the article
    (strict mode — no date = unknown age = skip).
    """
    if not pub_date_str:
        return False
    try:
        dt = dateutil_parser.parse(pub_date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= _cutoff_dt()
    except Exception:
        return False

# ─── Locations ───────────────────────────────────────────────────────────────

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

GLOBAL_QUERIES = [
    "breaking news",
    "world news today",
    "international news",
    "latest global news",
]

# ─── Dedup ───────────────────────────────────────────────────────────────────

def _is_new_title(title: str) -> bool:
    key = title.strip().lower()
    if key in _seen_titles:
        return False
    _seen_titles.add(key)
    return True

# ─── API param builders ───────────────────────────────────────────────────────

def _newsdata_params(query: str, country: str | None = None) -> dict:
    params = {
        "apikey":         newsdata_api_key,
        "language":       "en",
        "q":              query,
        "timeframe":      MAX_AGE_HOURS,   # dynamic — matches our window
        "prioritydomain": "top",
    }
    if country:
        params["country"] = country
    return params


def _gnews_params(query: str, country: str | None = None, max_results: int = 5) -> dict:
    cutoff = _cutoff_dt().strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "apikey":  gnews_key,
        "lang":    "en",
        "q":       query,
        "max":     max_results,
        "sortby":  "publishedAt",
        "from":    cutoff,
    }
    if country:
        params["country"] = country
    return params

# ─── Fetchers ─────────────────────────────────────────────────────────────────

def _fetch_newsdata(query: str, country: str | None = None, limit: int = 5) -> list[str]:
    titles = []
    try:
        res = requests.get(
            "https://newsdata.io/api/1/latest",
            params=_newsdata_params(query, country),
            timeout=15,
        ).json()
        for item in res.get("results", []):
            title    = (item.get("title") or "").strip()
            pub_date = item.get("pubDate") or item.get("publishedAt") or ""

            if not title:
                continue
            if not _is_fresh(pub_date):
                print(f"    ⏰ Skipped stale Newsdata article: {title[:60]} | date={pub_date}")
                continue
            if _is_new_title(title):
                titles.append(title)
            if len(titles) >= limit:
                break
    except Exception as e:
        print(f"  Newsdata error [{query}]:", e)
    return titles


def _fetch_gnews(query: str, country: str | None = None, limit: int = 5) -> list[str]:
    titles = []
    try:
        res = requests.get(
            "https://gnews.io/api/v4/search",
            params=_gnews_params(query, country, max_results=limit * 2),  # over-fetch to survive stale drops
            timeout=15,
        ).json()
        for item in res.get("articles", []):
            title    = (item.get("title") or "").strip()
            pub_date = item.get("publishedAt") or ""

            if not title:
                continue
            if not _is_fresh(pub_date):
                print(f"    ⏰ Skipped stale GNews article: {title[:60]} | date={pub_date}")
                continue
            if _is_new_title(title):
                titles.append(title)
            if len(titles) >= limit:
                break
    except Exception as e:
        print(f"  GNews error [{query}]:", e)
    return titles


def _fetch_latest_titles(query: str, country: str | None, limit: int = 5) -> list[str]:
    """Newsdata first, GNews for shortfall. All results guaranteed ≤ 48 h old."""
    titles = _fetch_newsdata(query, country, limit)
    if len(titles) < limit:
        titles += _fetch_gnews(query, country, limit - len(titles))
    return titles[:limit]

# ─── BBC scraper (with date filtering) ───────────────────────────────────────

def _bbc_article_pub_date(article_tag) -> str | None:
    """
    Try to extract a publish date from a BBC article card.
    BBC embeds dates in <time datetime="..."> tags inside the card.
    """
    time_tag = article_tag.find("time")
    if time_tag:
        return time_tag.get("datetime") or time_tag.get_text(strip=True) or None
    return None


async def scrape():
    """
    BBC headlines with freshness filtering.
    Only saves articles that have a <time> tag within the last 48 hours.
    Articles with no date are skipped (strict mode).
    """
    print("\n📰 Scraping BBC...")
    try:
        r = requests.get(bbc, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        saved = 0

        # Walk every <article> on the page
        for article_tag in soup.find_all("article"):
            pub_date = _bbc_article_pub_date(article_tag)

            if not _is_fresh(pub_date):
                # No date or too old — skip entirely
                continue

            # Try headline selectors inside this article
            headline_tag = (
                article_tag.select_one("h3")
                or article_tag.select_one("[data-testid='card-headline']")
                or article_tag.select_one("h2")
            )
            if not headline_tag:
                continue

            text = headline_tag.get_text(strip=True)
            if not text or len(text) <= 10:
                continue
            if not _is_new_title(text):
                continue

            article = {"tittle": text, "city": "India", "state": "India"}
            save_to_scrape_table(text, "India", "India")
            print(f"  BBC saved [{pub_date}]: {text}")
            await ai_itellengence(article)
            saved += 1

            if saved >= 10:
                break

        if saved == 0:
            print("  ⚠️  No fresh BBC headlines found (all older than 48 h or no <time> tags)")

    except Exception as e:
        print("Scrape error:", e)

# ─── DB helpers ───────────────────────────────────────────────────────────────

async def clear_all_tables():
    try:
        supabase.table("content_radar_news_scrape").delete().neq("tittle", "").execute()
        supabase.table("content_radar").delete().neq("tittle", "").execute()
        print("🧹 Tables cleared")
    except Exception as e:
        print("Cleanup error:", e)


def save_to_scrape_table(tittle: str, city: str, state: str):
    try:
        supabase.table("content_radar_news_scrape").upsert(
            {"tittle": tittle, "city": city, "state": state},
            on_conflict="tittle",
            ignore_duplicates=False,
        ).execute()
    except Exception as e:
        print(f"  Scrape table insert failed [{city or state or 'global'}]:", e)

# ─── AI processing ────────────────────────────────────────────────────────────

async def optimise_tittle(tittle: str) -> str:
    prompt = (
        "You are a NEWS TOPIC TAG generator.\n"
        "Convert the news headline into a 2–5 word topic label.\n"
        "RULES: Only the label. No punctuation. No explanation. No full sentences.\n\n"
        "EXAMPLES:\n"
        "Attack on Donald Trump\n"
        "Iran Israel Conflict\n"
        "US China Trade War\n"
        "Middle East Crisis\n"
    )
    try:
        res = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": tittle},
            ],
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("Optimise error:", e)
        return tittle


async def ai_itellengence(article: dict):
    print("AI running...")

    raw_tittle = article.get("tittle", "").strip()
    if not raw_tittle:
        return

    city  = article.get("city", "")  or ""
    state = article.get("state", "") or ""

    optimized = await optimise_tittle(raw_tittle)

    location_context = ""
    if city or state:
        parts = [p for p in [city, state] if p]
        location_context = f"\nLocation Context: {', '.join(parts)}"

    prompt = (
        "You are a Viral Content Strategist.\n\n"
        f"News:\n{optimized}{location_context}\n\n"
        "STEP 0: If low value → return SKIP\n\n"
        "STEP 1:\n"
        "1. Score\n2. Hooks\n3. Emotion\n4. Script\n5. Hashtags"
    )

    try:
        def call_model():
            return deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user",   "content": prompt},
                ],
            ).choices[0].message.content.strip()

        output = await asyncio.to_thread(call_model)

        if output.strip().upper() == "SKIP":
            print(f"  ⏭️  Skipped [{city or state or 'global'}]: {raw_tittle}")
            return

        supabase.table("content_radar").upsert(
            {
                "tittle":         optimized,
                "regular_tittle": raw_tittle,
                "summary":        output,
                "city":           city,
                "state":          state,
            },
            on_conflict="regular_tittle,city,state",
        ).execute()

        label = city or state or "global"
        print(f"  ✅ Saved → content_radar [{label}]: {optimized}")

    except Exception as e:
        print(f"  AI error [{city or state or 'global'}]:", e)

# ─── Section runners ──────────────────────────────────────────────────────────

async def fetch_india_city_news():
    print("\n🇮🇳 Fetching India city news (1 article per city)...")

    for loc in INDIA_LOCATIONS:
        city  = loc["city"]
        state = loc["state"]
        label = f"{city}, {state}"
        print(f"  📍 {label}")

        query  = f"{city} {state}".strip()
        titles = _fetch_latest_titles(query, country="in", limit=1)

        if titles:
            title   = titles[0]
            article = {"tittle": title, "city": city, "state": state}
            save_to_scrape_table(title, city, state)
            print(f"  Saved [{city}]:", title)
            await ai_itellengence(article)
        else:
            print(f"  ⚠️  No fresh article found for {label}")

        await asyncio.sleep(1)


async def fetch_india_state_news():
    print("\n🗺️  Fetching India state news (5 per state)...")

    for loc in INDIA_STATES:
        state  = loc["state"]
        city   = state
        label  = state
        print(f"  📍 {label}")

        titles = _fetch_latest_titles(state, country="in", limit=5)

        for title in titles:
            article = {"tittle": title, "city": city, "state": state}
            save_to_scrape_table(title, city, state)
            print(f"  Saved [{state}]:", title)
            await ai_itellengence(article)

        if not titles:
            print(f"  ⚠️  No fresh articles found for {label}")

        await asyncio.sleep(1)


async def fetch_global_news():
    print("\n🌍 Fetching global news (1 per query)...")

    for query in GLOBAL_QUERIES:
        print(f"  🌐 {query}")

        titles = _fetch_latest_titles(query, country=None, limit=1)

        if titles:
            title = titles[0]
            save_to_scrape_table(title, "global", "global")
            print(f"  Saved [global]:", title)
            await ai_itellengence({"tittle": title, "city": "global", "state": "global"})
        else:
            print(f"  ⚠️  No fresh article found for query: {query}")

        await asyncio.sleep(1)


async def cycle():
    global _seen_titles
    _seen_titles = set()

    print(f"\n🕐 Cycle start — accepting articles from last {MAX_AGE_HOURS} hours")
    print(f"   Cutoff: {_cutoff_dt().strftime('%Y-%m-%d %H:%M UTC')}")

    print("\n🧹 Clearing old data...")
    await clear_all_tables()

    await fetch_india_city_news()
    await fetch_india_state_news()
    await fetch_global_news()
    await scrape()

    print("\n✅ Full cycle complete — next run in 24 hours")


async def main():
    while True:
        try:
            await cycle()
        except Exception as e:
            print("❌ Cycle error:", e)

        await asyncio.sleep(24 * 60 * 60)


if __name__ == "__main__":
    asyncio.run(main())