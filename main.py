import os
import time
import asyncio
import requests
import re
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from openai import OpenAI
from supabase import create_client
import schedule  

load_dotenv()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

TABLE_NAME = "content_radar"
PER_BUCKET_TARGET = 10   
AI_CONCURRENCY = 4       

# How old an article is allowed to be before we drop it client-side.
# NewsAPI's free "Developer" plan doesn't give you same-day articles — it
# has its own delay ceiling (can be days to weeks). We no longer request a
# "from" window (that was causing totalResults: 0 since the window fell
# inside the plan's blackout period). Instead we take whatever NewsAPI
# hands back sorted by publishedAt, and filter/log freshness ourselves.
MAX_ARTICLE_AGE_DAYS = 30

NEWSAPI_GLOBAL_SOURCES = (
    "bbc-news,reuters,associated-press,al-jazeera-english,"
    "the-guardian-uk,deutsche-welle,france-24,cnn,bloomberg,the-hindu"
)

NEWSAPI_INDIA_DOMAINS = (
    "ndtv.com,indiatoday.in,thehindu.com,hindustantimes.com,"
    "timesofindia.indiatimes.com,economictimes.indiatimes.com,"
    "indianexpress.com,livemint.com,business-standard.com"
)


def _clean_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 300:
        text = text[:300].strip()
    return text


def _safe_get(url, params=None, tag=""):
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp, None
        body = resp.text[:300].replace("\n", " ")
        msg = f"HTTP {resp.status_code} — {body}"
        print(f"   [{tag}] ❌ {msg}")
        return None, msg
    except requests.exceptions.Timeout:
        msg = "Request timed out"
        print(f"   [{tag}] ❌ {msg}")
        return None, msg
    except requests.exceptions.ConnectionError as e:
        msg = f"Connection error: {e}"
        print(f"   [{tag}] ❌ {msg}")
        return None, msg
    except Exception as e:
        msg = f"Unexpected error: {e}"
        print(f"   [{tag}] ❌ {msg}")
        return None, msg


def is_table_empty() -> bool:
    """Returns True if content_radar currently has 0 rows."""
    try:
        res = supabase.table(TABLE_NAME).select("id", count="exact").limit(1).execute()
        count = res.count if res.count is not None else len(res.data)
        return count == 0
    except Exception as e:
        print(f"⚠️  Could not check if '{TABLE_NAME}' is empty ({e}) — assuming NOT empty")
        return False


def clear_supabase_table():
    """Delete every row from content_radar before pushing fresh data."""
    try:
        supabase.table(TABLE_NAME).delete().neq("id", 0).execute()
        print(f"🗑️  Cleared all existing rows from '{TABLE_NAME}'")
    except Exception as e:
        print(f"⚠️  Failed to clear table '{TABLE_NAME}': {e}")


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for r in rows:
        key = r["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _parse_published_at(published_at: str):
    if not published_at:
        return None
    try:
        return datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_recent_enough(published_at: str, max_age_days: int = MAX_ARTICLE_AGE_DAYS) -> bool:
    pub = _parse_published_at(published_at)
    if pub is None:
        return True  # don't drop on parse failure
    return (datetime.now(timezone.utc) - pub) <= timedelta(days=max_age_days)


_NON_NEWS_MARKERS = (
    "sponsored", "advertisement", "promoted", "partner content",
    "[removed]", "press release", "coupon", "deal of the day",
)


def _is_real_news(title: str, description: str) -> bool:
    """Basic quality gate to filter out sponsored posts, listicles, and empty stubs."""
    t = (title or "").strip().lower()
    d = (description or "").strip()
    if not t or len(t) <= 10:
        return False
    if any(marker in t for marker in _NON_NEWS_MARKERS):
        return False
    if any(marker in d.lower() for marker in _NON_NEWS_MARKERS):
        return False
    if not d or len(d) < 30:
        # real news articles almost always ship a real description;
        # a missing/very short one is usually a stub or listicle entry
        return False
    return True


def _log_freshness(tag: str, published_ats: list[str]):
    """Print the oldest/newest publishedAt in a batch so plan delay is visible at a glance."""
    parsed = [p for p in (_parse_published_at(x) for x in published_ats) if p]
    if not parsed:
        return
    newest, oldest = max(parsed), min(parsed)
    age_hours = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
    print(f"   🕒 [{tag}] freshest article: {newest.isoformat()} ({age_hours:.1f}h old) | oldest: {oldest.isoformat()}")


def fetch_newsapi(
    sources: str = None, query: str = None, domains: str = None,
    max_articles: int = 100, region_label: str = "Global",
) -> list[dict]:
    """
    Uses NewsAPI's /v2/everything endpoint.

    NOTE: /v2/top-headlines is restricted on the free "Developer" plan —
    calling it from a live server (not localhost) silently returns
    status "ok" with totalResults 0, regardless of the country/sources
    params passed. /v2/everything does not have that restriction, so we
    use it for both national and global pulls, filtering via `q` and/or
    `domains`/`sources` instead of `country`.

    IMPORTANT: we intentionally do NOT send a "from" param. The free plan
    has its own delay ceiling on the newest article it will return (can be
    days to weeks behind real-time). Requesting from=now-24h landed inside
    that blackout window and produced totalResults: 0 on every call. We
    let sortBy=publishedAt surface the freshest articles the plan actually
    has, then filter/log freshness client-side instead.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        print("   ⚠️  NEWSAPI_KEY not set — skipping NewsAPI fetch")
        return []

    endpoint = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "pageSize": min(max_articles, 100),
        "sortBy": "publishedAt",
        "language": "en",
    }
    if sources:
        params["sources"] = sources
    if domains:
        params["domains"] = domains
    if query:
        params["q"] = query
    if not sources and not domains and not query:
        params["q"] = "news"  

    resp, err = _safe_get(endpoint, params=params, tag="NEWSAPI")
    if resp is None:
        return []

    data = resp.json()

    if data.get("status") == "error":
        print(f"   [NEWSAPI] ❌ {data.get('code', 'error')}: {data.get('message', 'unknown error')}")
        return []

    articles = data.get("articles", [])
    _log_freshness("NEWSAPI:" + region_label, [a.get("publishedAt", "") for a in articles])

    rows = []
    skipped_stale = 0
    skipped_non_news = 0
    for art in articles:
        title = _clean_title(art.get("title", "") or "")
        description = art.get("description", "") or ""
        source = art.get("source", {}).get("name", "NewsAPI")
        published_at = art.get("publishedAt", "")

        if not _is_real_news(title, description):
            skipped_non_news += 1
            continue
        if not _is_recent_enough(published_at):
            skipped_stale += 1
            continue
        rows.append({"region": region_label, "source": source, "title": title})

    if skipped_non_news:
        print(f"   🚫 [{region_label}] dropped {skipped_non_news} non-news/low-quality entries")
    if skipped_stale:
        print(f"   ⏳ [{region_label}] dropped {skipped_stale} articles older than {MAX_ARTICLE_AGE_DAYS}d")

    print(f"   📡 NewsAPI → {region_label}: {len(rows)} articles (totalResults={data.get('totalResults', '?')})")
    return rows



def fetch_gnews(
    query: str = "top news", country: str = "in", lang: str = "en",
    max_articles: int = 10, region_label: str = "National",
) -> list[dict]:
    api_key = os.getenv("GNEWS_KEY", "")
    if not api_key:
        print("   ⚠️  GNEWS_KEY not set — skipping GNews fetch")
        return []

    endpoint = "https://gnews.io/api/v4/top-headlines"
    params = {
        "token": api_key, "lang": lang, "country": country,
        "max": min(max_articles, 10), "q": query,
    }

    resp, err = _safe_get(endpoint, params=params, tag="GNEWS")
    if resp is None:
        return []

    data = resp.json()

    if "errors" in data:
        print(f"   [GNEWS] ❌ {data.get('errors')}")
        return []

    articles = data.get("articles", [])
    _log_freshness("GNEWS:" + region_label, [a.get("publishedAt", "") for a in articles])

    rows = []
    skipped_non_news = 0
    for art in articles:
        title = _clean_title(art.get("title", "") or "")
        description = art.get("description", "") or ""
        source = art.get("source", {}).get("name", "GNews")
        if not _is_real_news(title, description):
            skipped_non_news += 1
            continue
        rows.append({"region": region_label, "source": source, "title": title})

    if skipped_non_news:
        print(f"   🚫 [{region_label}] dropped {skipped_non_news} non-news/low-quality entries")
    print(f"   📡 GNews  → {region_label} ({query}): {len(rows)} articles")
    return rows



def get_national_news() -> list[dict]:
    print("\n📰 Fetching National (India) News...")
    print("─" * 50)

    rows = []
    rows += fetch_newsapi(query="India", domains=NEWSAPI_INDIA_DOMAINS, max_articles=30, region_label="National")
    rows += fetch_gnews(query="India", country="in", lang="en", max_articles=10, region_label="National")

    unique = _dedupe(rows)
    print(f"\n  ✅ National: {len(unique)} unique titles collected")
    return unique



def get_global_news() -> list[dict]:
    print("\n🌍 Fetching Global News...")
    print("─" * 50)

    rows = []
    rows += fetch_newsapi(sources=NEWSAPI_GLOBAL_SOURCES, max_articles=30, region_label="Global")
    rows += fetch_gnews(query="world news", country="us", lang="en", max_articles=10, region_label="Global")

    unique = _dedupe(rows)
    print(f"\n  ✅ Global: {len(unique)} unique titles collected")
    return unique


def pick_top_n_titles(rows: list[dict], n: int = PER_BUCKET_TARGET) -> list[str]:
    titles = [r["title"] for r in rows if r.get("title")]
    titles = list(dict.fromkeys(titles))        
    titles.sort(key=len, reverse=True)            
    return titles[:n]



async def optimise_title(title: str) -> str:
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
        res = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": title},
                ],
            )
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("  optimise_title error:", e)
        return title


async def ai_intelligence(raw_title: str, category: str = ""):
    print(f"  🤖 AI processing: {raw_title[:70]}")
    optimized = await optimise_title(raw_title)
    location_context = f"\nCategory Context: {category}" if category else ""
    prompt = (
        "You are a Viral Content Strategist.\n\n"
        f"News:\n{optimized}{location_context}\n\n"
        "STEP 0: If low value → return SKIP\n\n"
        "STEP 1:\n"
        "1. Score\n2. Hooks\n3. Emotion\n4. Script\n5. Hashtags"
    )
    try:
        res = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user", "content": prompt},
                ],
            )
        )
        output = res.choices[0].message.content.strip()

        if output.strip().upper() == "SKIP":
            print(f"   Skipped: {raw_title[:70]}")
            return

        supabase.table(TABLE_NAME).insert({
            "tittle": optimized,
            "regular_tittle": raw_title,
            "category": category,
        }).execute()
        print(f"  ✅ Saved → {TABLE_NAME} [{category}]: {optimized}")
    except Exception as e:
        print(f"  AI error:", e)


async def _process_bucket(titles: list[str], category: str, semaphore: asyncio.Semaphore):
    async def _run(t):
        async with semaphore:
            await ai_intelligence(t, category)

    await asyncio.gather(*(_run(t) for t in titles))


async def push_to_db(national_rows: list[dict], global_rows: list[dict], per_bucket: int = PER_BUCKET_TARGET):
    """Wipe content_radar, pick top titles per bucket, run AI intelligence, insert."""
    clear_supabase_table()

    semaphore = asyncio.Semaphore(AI_CONCURRENCY)

    nat_titles = pick_top_n_titles(national_rows, n=per_bucket)
    print(f"\n📰 National → pushing {len(nat_titles)} titles (target {per_bucket})")
    await _process_bucket(nat_titles, "national", semaphore)

    glob_titles = pick_top_n_titles(global_rows, n=per_bucket)
    print(f"\n🌍 International → pushing {len(glob_titles)} titles (target {per_bucket})")
    await _process_bucket(glob_titles, "international", semaphore)

    print("\n🎉 DB push complete!")



def run():
    print(f"\n{'=' * 70}")
    print(f"🚀 Pipeline run starting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    national_rows = get_national_news()
    global_rows = get_global_news()

    asyncio.run(push_to_db(national_rows, global_rows, per_bucket=PER_BUCKET_TARGET))

    print(f"\n🎉 Pipeline run complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



def start_scheduler():
    if is_table_empty():
        print(f"▶️  '{TABLE_NAME}' is empty — running pipeline once immediately.")
        run()
    else:
        print(f"⏭️  '{TABLE_NAME}' already has data — skipping immediate run, "
              f"waiting for the 00:00 schedule.")

    schedule.every().day.at("00:00").do(run)
    print("\n⏰ Scheduler active — pipeline will run every night at 00:00 local time.")
    print("   (Leave this process running, e.g. inside tmux/screen or as a service.)")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()