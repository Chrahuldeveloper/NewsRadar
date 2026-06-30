import os
import time
import asyncio
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import urllib.parse
import random
import feedparser
import json
from datetime import datetime, timedelta
import re

from openai import OpenAI
from supabase import create_client

load_dotenv()

OUTPUT_DIR = Path("news_output")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── Supabase / DeepSeek clients ─────────────────────────────────────────────
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

TABLE_NAME = "content_radar"
PER_BUCKET_TARGET = 10          # every state / national / global must have exactly 10
AI_CONCURRENCY = 4               # how many DeepSeek calls run in parallel

# ─── Regional language domains (for auto-detect translation) ────────────────
REGIONAL_LANGUAGE_DOMAINS = {
    "sakshi.com", "andhrajyothy.com", "tv9telugu.com", "ntnews.com",
    "dailythanthi.com", "dinamalar.com", "thanthitv.com",
    "prajavani.net", "udayavani.com", "kannadaprabha.com",
    "tv9kannada.com", "publictv.in",
    "manoramaonline.com", "mathrubhumi.com", "asianetnews.com", "janamtv.com",
    "lokmat.com", "maharashtratimes.com", "esakal.com",
    "divyabhaskar.co.in",
    "patrika.com", "bhaskar.com", "jagran.com", "amarujala.com",
    "livehindustan.com", "prabhatkhabar.com", "naidunia.com",
    "ajitjalandhar.com", "jagbani.punjabkesari.in",
    "bartamanpatrika.com", "tv9bangla.com",
    "sambad.in", "dharitri.com", "kanaknews.com", "nandighoshatv.com",
    "dy365.in", "niyomiyabarta.com",
}

# ─── Global news RSS feeds (international sources) ──────────────────────────
GLOBAL_RSS_FEEDS = {
    "BBC World":          "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters World":      "https://feeds.reuters.com/reuters/worldnews",
    "Al Jazeera":         "https://www.aljazeera.com/xml/rss/all.xml",
    "AP News":            "https://rsshub.app/apnews/world",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "DW News":            "https://rss.dw.com/xml/rss-en-world",
    "France 24":          "https://www.france24.com/en/rss",
    "CNN World":          "http://rss.cnn.com/rss/edition_world.rss",
    "NPR World":          "https://feeds.npr.org/1004/rss.xml",
    "SCMP":               "https://www.scmp.com/rss/91/feed",
    "NHK World":          "https://www3.nhk.or.jp/rss/news/cat0.xml",
    "Times of India":     "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
}

# ─── NewsAPI.org global sources (set NEWSAPI_KEY in .env) ───────────────────
NEWSAPI_GLOBAL_SOURCES = (
    "bbc-news,reuters,associated-press,al-jazeera-english,"
    "the-guardian-uk,deutsche-welle,france-24,cnn,bloomberg,the-hindu"
)

# ─── National Indian RSS feeds (fallback / supplement) ──────────────────────
NATIONAL_RSS_FEEDS = {
    "NDTV":              "https://feeds.feedburner.com/ndtvnews-top-stories",
    "India Today":       "https://www.indiatoday.in/rss/home",
    "The Hindu":         "https://www.thehindu.com/feeder/default.rss",
    "Hindustan Times":   "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "Times of India":    "https://timesofindia.indiatimes.com/rssfeeds/1221656.cms",
    "Economic Times":    "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    "Indian Express":    "https://indianexpress.com/section/india/feed/",
    "Livemint":          "https://www.livemint.com/rss/news",
    "Business Standard": "https://www.business-standard.com/rss/home_page_top_stories.rss",
    "NewsLaundry":       "https://www.newslaundry.com/feed",
    "The Wire":          "https://thewire.in/feed",
}

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _needs_translation(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in REGIONAL_LANGUAGE_DOMAINS)


def _safe_get(url, params=None, tag=""):
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp, None
        body = resp.text[:300].replace("\n", " ")
        msg  = f"HTTP {resp.status_code} — {body}"
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


# ─── Translation config ──────────────────────────────────────────────────────

_GT_ENDPOINTS = [
    "https://translate.googleapis.com/translate_a/single",
    "https://translate.google.com/translate_a/single",
]

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

_translate_session = requests.Session()
_endpoint_idx      = 0

_INDIAN_SCRIPT_RANGES = [
    (0x0900, 0x097F), (0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF),
    (0x0B00, 0x0B7F), (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF),
    (0x0D00, 0x0D7F), (0x0D80, 0x0DFF), (0x0600, 0x06FF),
]


def _is_non_english(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _INDIAN_SCRIPT_RANGES:
            if lo <= cp <= hi:
                return True
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results and results[0].lang == "en" and results[0].prob >= 0.90:
            return False
        if results and results[0].lang != "en":
            return True
    except Exception:
        pass
    return False


def _gt_translate_raw(text: str, retries: int = 5) -> str | None:
    global _endpoint_idx
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]", "", text).strip()
    if not text:
        return None
    if len(text) > 500:
        text = text[:500]

    base_params = {"client": "gtx", "sl": "auto", "tl": "en", "dt": "t"}

    for attempt in range(retries):
        url     = _GT_ENDPOINTS[(_endpoint_idx + attempt) % len(_GT_ENDPOINTS)]
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        try:
            if attempt % 2 == 0:
                resp = _translate_session.get(
                    url, params={**base_params, "q": text}, headers=headers, timeout=15,
                )
            else:
                resp = _translate_session.post(
                    url, params=base_params, data={"q": text},
                    headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15,
                )

            if resp.status_code == 200:
                try:
                    data   = resp.json()
                    result = "".join(
                        part[0] for part in data[0]
                        if isinstance(part, list) and part and part[0]
                    ).strip()
                    if result:
                        _endpoint_idx += 1
                        return result
                    time.sleep(0.5 * (attempt + 1))
                    continue
                except (ValueError, KeyError, IndexError, TypeError):
                    time.sleep(1)
                    continue
            elif resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(2, 5)
                time.sleep(wait)
            elif resp.status_code in (500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except requests.exceptions.Timeout:
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.ConnectionError:
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(1)

    _endpoint_idx += 1
    return None


def translate_to_english(title: str) -> str:
    if not title or not title.strip():
        return title
    if not _is_non_english(title):
        return title
    result = _gt_translate_raw(title)
    return result if result else title


def translate_titles(titles: list[str], source_label: str = "") -> list[str]:
    translated    = []
    total         = len(titles)
    fails         = 0
    skipped_en    = 0
    print(f"   🌐 Translating {total} titles to English...", flush=True)

    for i, title in enumerate(titles):
        needs_tr = _is_non_english(title)
        if not needs_tr:
            translated.append(title)
            skipped_en += 1
        else:
            result = _gt_translate_raw(title)
            if result:
                translated.append(result)
            else:
                translated.append(title)
                fails += 1
                print(f"   ⚠️  Failed to translate [{fails}]: {title[:60]!r}")

        if needs_tr:
            base = 0.15 if fails == 0 else 0.5
            time.sleep(base + random.uniform(0, 0.25))

        if (i + 1) % 20 == 0:
            print(f"      ... {i + 1}/{total} done  "
                  f"(skipped-English: {skipped_en}, failed: {fails})")

    status = "✅" if fails == 0 else "⚠️ "
    print(f"   {status} Translation done [{source_label}] — "
          f"{total - skipped_en - fails} translated, "
          f"{skipped_en} already English, "
          f"{fails} failed")
    return translated


def _clean_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 300:
        cut = text[:300]
        for sep in ("।", ".", "!", "?", ":", "|"):
            idx = cut.rfind(sep)
            if idx > 80:
                text = cut[: idx + 1].strip()
                break
        else:
            text = cut.strip()
    return text


# ════════════════════════════════════════════════════════════════════════════
# SCRAPING CORE
# ════════════════════════════════════════════════════════════════════════════

def _extract_titles_from_soup(soup: BeautifulSoup, selector: str) -> list[str]:
    titles = []

    def _text(el):
        return _clean_title(el.get_text(strip=True))

    if selector and selector.lower() not in ("", "nan", "none"):
        for el in soup.select(selector):
            text = _text(el)
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles))

    for el in soup.select("h1, h2, h3"):
        text = _text(el)
        if text and len(text) > 10:
            titles.append(text)

    return list(dict.fromkeys(titles))


def scrape_titles(url: str, selector: str = None) -> list[str]:
    resp, _ = _safe_get(url, tag="SCRAPE")
    if resp is None:
        return []

    soup = BeautifulSoup(resp.content, "html.parser")

    if "sakshi.com" in url:
        titles = []
        for el in soup.select("h2, h3, a, .title, .story-title, .news-title"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 25:
                titles.append(text)
        return list(dict.fromkeys(titles))

    if "ndtv.com" in url:
        titles = []
        for el in soup.select(".news_Itm-cont h2, .NwsLstPg_ttl, .pst-by h2"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    if "thehindu.com" in url:
        titles = []
        for el in soup.select("h2.title, h3.title, .story-card-news h3, a.story-card__title"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    if "indiatoday.in" in url:
        titles = []
        for el in soup.select(".view-content h3, .field--name-title, .story__title"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    if "hindustantimes.com" in url:
        titles = []
        for el in soup.select("h3.hdg3, .storyShortDetail h3, .story-title"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    if "timesofindia" in url:
        titles = []
        for el in soup.select("figcaption, .W_tti, .KafIv, .uwU81"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    if "economictimes" in url:
        titles = []
        for el in soup.select(".eachStory h3, .story-box h4, .artTitle"):
            text = _clean_title(el.get_text(strip=True))
            if text and len(text) > 10:
                titles.append(text)
        return list(dict.fromkeys(titles)) or _extract_titles_from_soup(soup, selector)

    return _extract_titles_from_soup(soup, selector)


# ════════════════════════════════════════════════════════════════════════════
# RSS FEED COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def fetch_rss_titles(feed_url: str, max_items: int = 50) -> list[str]:
    try:
        feed = feedparser.parse(feed_url)
        titles = []
        for entry in feed.entries[:max_items]:
            title = _clean_title(entry.get("title", ""))
            if title and len(title) > 10:
                titles.append(title)
        return list(dict.fromkeys(titles))
    except Exception as e:
        print(f"   ⚠️  RSS parse error ({feed_url[:60]}): {e}")
        return []


def get_rss_news(feed_map: dict, region_label: str) -> list[dict]:
    rows = []
    for source, rss_url in feed_map.items():
        print(f"   📡 RSS  → {source}", end=" ", flush=True)
        titles = fetch_rss_titles(rss_url)
        print(f"({len(titles)} titles)")
        for title in titles:
            rows.append({"region": region_label, "source": source, "title": title})
        time.sleep(0.5)
    return rows


# ════════════════════════════════════════════════════════════════════════════
# NEWSAPI COLLECTION  (global only)
# ════════════════════════════════════════════════════════════════════════════

def fetch_newsapi(
    sources: str = None, query: str = None, country: str = None,
    category: str = None, max_articles: int = 100, region_label: str = "Global",
) -> list[dict]:
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        print("   ⚠️  NEWSAPI_KEY not set — skipping NewsAPI fetch")
        return []

    base = "https://newsapi.org/v2"
    if country or category:
        endpoint = f"{base}/top-headlines"
        params   = {"apiKey": api_key, "pageSize": min(max_articles, 100)}
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if sources:
            params["sources"] = sources
        if query:
            params["q"] = query
    else:
        endpoint = f"{base}/everything"
        from_dt  = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        params   = {
            "apiKey": api_key, "pageSize": min(max_articles, 100),
            "sortBy": "publishedAt", "from": from_dt, "language": "en",
        }
        if sources:
            params["sources"] = sources
        if query:
            params["q"] = query

    resp, err = _safe_get(endpoint, params=params, tag="NEWSAPI")
    if resp is None:
        return []

    data     = resp.json()
    articles = data.get("articles", [])
    rows     = []
    for art in articles:
        title  = _clean_title(art.get("title", "") or "")
        source = art.get("source", {}).get("name", "NewsAPI")
        if title and len(title) > 10 and title.lower() != "[removed]":
            rows.append({"region": region_label, "source": source, "title": title})

    print(f"   📡 NewsAPI → {region_label}: {len(rows)} articles")
    return rows


# ════════════════════════════════════════════════════════════════════════════
# GNEWS API COLLECTION  (global only — free tier available)
# ════════════════════════════════════════════════════════════════════════════

def fetch_gnews(
    query: str = "top news", country: str = "in", lang: str = "en",
    max_articles: int = 10, region_label: str = "National",
) -> list[dict]:
    api_key = os.getenv("GNEWS_KEY", "")
    if not api_key:
        print("   ⚠️  GNEWS_KEY not set — skipping GNews fetch")
        return []

    endpoint = "https://gnews.io/api/v4/top-headlines"
    params   = {
        "token": api_key, "lang": lang, "country": country,
        "max": min(max_articles, 10), "q": query,
    }

    resp, err = _safe_get(endpoint, params=params, tag="GNEWS")
    if resp is None:
        return []

    data     = resp.json()
    articles = data.get("articles", [])
    rows     = []
    for art in articles:
        title  = _clean_title(art.get("title", "") or "")
        source = art.get("source", {}).get("name", "GNews")
        if title and len(title) > 10:
            rows.append({"region": region_label, "source": source, "title": title})

    print(f"   📡 GNews  → {region_label} ({query}): {len(rows)} articles")
    return rows


# ════════════════════════════════════════════════════════════════════════════
# NATIONAL NEWS  (CSV scraping + RSS only — no APIs)
# ════════════════════════════════════════════════════════════════════════════

def get_national_news(df_national: pd.DataFrame) -> list[dict]:
    print("\n📰 Fetching National News...")
    print("─" * 50)

    rows = []

    for _, row in df_national.iterrows():
        source   = str(row.get("Source Name", "")).strip()
        url      = str(row.get("url",         "")).strip()
        selector = str(row.get("Selector",    "")).strip()

        if not url or url.lower() in ("nan", "none", ""):
            print(f"\n  ⚠️  {source} — no URL, skipping")
            continue

        print(f"\n  📌 {source}")
        print("  " + "─" * 40)
        print("   → scraping ...", end=" ", flush=True)

        titles = scrape_titles(url, selector)
        print(f"{len(titles)} titles" if titles else "⚠️  none found")

        for title in titles:
            rows.append({"region": "National", "source": source, "title": title})

        time.sleep(1)

    print("\n  📡 Supplementing with National RSS feeds...")
    rows += get_rss_news(NATIONAL_RSS_FEEDS, region_label="National")

    seen   = set()
    unique = []
    for r in rows:
        key = r["title"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"\n  ✅ National: {len(unique)} unique titles collected")
    return unique


# ════════════════════════════════════════════════════════════════════════════
# REGIONAL NEWS  (CSV scraping only — no APIs)
# ════════════════════════════════════════════════════════════════════════════

def get_regional_news(df_regional: pd.DataFrame) -> list[dict]:
    print("\n🗺️  Fetching Regional News...")
    print("─" * 50)

    rows = []

    for _, row in df_regional.iterrows():
        state           = str(row.get("State",          "")).strip()
        source          = str(row.get("Source",         "")).strip()
        url             = str(row.get("URL",            "")).strip()
        selector        = str(row.get("Selector",       "")).strip()
        force_translate = str(row.get("ForceTranslate", "")).strip().lower() == "yes"

        if not url or url.lower() in ("nan", "none", ""):
            print(f"\n  ⚠️  {state} → {source} — no URL, skipping")
            continue

        force_tr  = _needs_translation(url) or force_translate
        print(f"\n  📌 {state} → {source}")
        print("  " + "─" * 40)
        print("   → scraping ...", end=" ", flush=True)

        raw_titles = scrape_titles(url, selector)
        print(f"{len(raw_titles)} titles" if raw_titles else "⚠️  none found")

        if not raw_titles:
            time.sleep(1)
            continue

        any_non_english = force_tr or any(_is_non_english(t) for t in raw_titles)
        final_titles = translate_titles(raw_titles, source_label=source) if any_non_english else raw_titles

        for title in final_titles:
            rows.append({"region": state, "source": source, "title": title})

        time.sleep(1)

    seen   = set()
    unique = []
    for r in rows:
        key = (r["region"], r["title"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"\n  ✅ Regional: {len(unique)} unique titles collected")
    return unique


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL NEWS  (RSS + APIs only — no web scraping)
# ════════════════════════════════════════════════════════════════════════════

def get_global_news() -> list[dict]:
    print("\n🌍 Fetching Global News...")
    print("─" * 50)

    rows = []

    print("\n  📡 Global RSS feeds...")
    rows += get_rss_news(GLOBAL_RSS_FEEDS, region_label="Global")

    print("\n  📡 Global NewsAPI...")
    newsapi_rows = fetch_newsapi(
        sources=NEWSAPI_GLOBAL_SOURCES, max_articles=100, region_label="Global",
    )
    for r in newsapi_rows:
        rows.append({"country": r["source"], "source": r["source"], "title": r["title"]})

    print("\n  📡 GNews (world)...")
    gnews_rows = fetch_gnews(
        query="world news", country="us", lang="en", max_articles=10, region_label="Global",
    )
    for r in gnews_rows:
        rows.append({"country": "Global", "source": r["source"], "title": r["title"]})

    seen   = set()
    unique = []
    for r in rows:
        key = r["title"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"\n  ✅ Global: {len(unique)} unique titles collected")
    return unique


# ════════════════════════════════════════════════════════════════════════════
# COMMON-HEADLINE CLUSTERING  (TF-IDF + cosine similarity — no embedding model)
# ════════════════════════════════════════════════════════════════════════════

def _cluster_titles(df: pd.DataFrame, threshold: float = 0.45) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    titles = df["title"].tolist()
    n = len(titles)
    if n < 2:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(titles)
    except ValueError:
        return []

    cos_sim  = cosine_similarity(tfidf_matrix)
    assigned = [False] * n
    clusters = []

    connectivity = [(int((cos_sim[i] >= threshold).sum()) - 1, i) for i in range(n)]
    connectivity.sort(reverse=True)

    for _, i in connectivity:
        if assigned[i]:
            continue

        similar_indices = [
            j for j in range(n)
            if not assigned[j] and i != j and cos_sim[i][j] >= threshold
        ]

        if not similar_indices:
            continue

        cluster_indices = [i] + similar_indices
        for idx in cluster_indices:
            assigned[idx] = True

        sources        = df.iloc[cluster_indices]["source"].tolist()
        cluster_titles = df.iloc[cluster_indices]["title"].tolist()
        unique_sources = list(dict.fromkeys(sources))
        orig_indices   = df.iloc[cluster_indices].index.tolist()

        clusters.append({
            "canonical_title": cluster_titles[0],
            "sources":         " | ".join(unique_sources),
            "unique_sources":  len(unique_sources),
            "repeat_count":    len(cluster_indices),
            "matched_titles":  " ||| ".join(cluster_titles),
            "orig_indices":    orig_indices,
        })

    return sorted(clusters, key=lambda x: (x["unique_sources"], x["repeat_count"]), reverse=True)


def _print_top(clusters: list[dict], label: str, top_n: int):
    print(f"\n   🔥 Top {top_n} Hot Topics [{label}]:")
    print("   " + "─" * 60)
    for rank, c in enumerate(clusters[:top_n], 1):
        print(f"\n   #{rank} [{c['repeat_count']}x | {c['unique_sources']} sources]")
        print(f"   {c['canonical_title'][:90]}")
        print(f"   📰 {c['sources'][:120]}")


def find_common_news(national_csv: Path, threshold: float = 0.45, top_n: int = 10):
    print("\n🔍 Finding Common Headlines Across Sources...")
    print("─" * 50)

    def _load(path, dataset_label):
        if not Path(path).exists():
            print(f"   ⚠️  {path} not found — skipping")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        if "source name" in df.columns:
            df = df.rename(columns={"source name": "source"})
        df = df[["source", "title"]].dropna(subset=["title"])
        df["title"]   = df["title"].astype(str).str.strip()
        df["source"]  = df["source"].astype(str).str.strip()
        df["dataset"] = dataset_label
        return df.reset_index(drop=True)

    df_nat = _load(national_csv, "national")

    if not df_nat.empty:
        print(f"\n   📰 National : {len(df_nat)} titles")
        print(f"\n   [1/2] Clustering NATIONAL ({len(df_nat)} titles) ...")
        nat_clusters = _cluster_titles(df_nat, threshold)
        _print_top(nat_clusters, "National", top_n)
    else:
        print("\n   ❌ National CSV empty — nothing to cluster.")

    global_csv = OUTPUT_DIR / "global_titles.csv"
    if global_csv.exists():
        df_glob = pd.read_csv(global_csv)
        df_glob.columns = df_glob.columns.str.strip().str.lower()
        df_glob = df_glob.rename(columns={"country": "source"})
        df_glob = df_glob[["source", "title"]].dropna(subset=["title"])
        df_glob["title"]  = df_glob["title"].astype(str).str.strip()
        df_glob["source"] = df_glob["source"].astype(str).str.strip()
        df_glob = df_glob.reset_index(drop=True)

        print(f"\n   [2/2] Clustering GLOBAL ({len(df_glob)} titles) ...")
        glob_clusters = _cluster_titles(df_glob, threshold)
        _print_top(glob_clusters, "Global", top_n)
    else:
        print("\n   ⚠️  global_titles.csv not found — skipping global clustering")


# ════════════════════════════════════════════════════════════════════════════
# PICK EXACTLY N TITLES PER BUCKET (clustering first, padded to N)
# ════════════════════════════════════════════════════════════════════════════

def pick_top_n_titles(df: pd.DataFrame, n: int = PER_BUCKET_TARGET, threshold: float = 0.45) -> list[str]:
    """
    Returns exactly `n` titles for a bucket (a state, National, or Global):
      1. Cluster on TF-IDF similarity, take the canonical title of the
         strongest clusters first (most repeated across sources = most
         "hot"/common).
      2. If there aren't enough clusters to reach n, pad with the remaining
         un-clustered titles (longest/most informative first) until we hit n.
      3. If there still aren't enough raw titles to reach n, return whatever
         is available (can't invent titles that don't exist).
    """
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 10].reset_index(drop=True)

    if df.empty:
        return []

    chosen = []
    used_idx = set()

    if len(df) >= 2:
        clusters = _cluster_titles(df, threshold)
        for c in clusters:
            if len(chosen) >= n:
                break
            chosen.append(c["canonical_title"])
            used_idx.update(c["orig_indices"])

    if len(chosen) < n:
        leftover = df[~df.index.isin(used_idx)].copy()
        leftover["len"] = leftover["title"].str.len()
        leftover = leftover.sort_values("len", ascending=False)
        for title in leftover["title"].tolist():
            if len(chosen) >= n:
                break
            if title not in chosen:
                chosen.append(title)

    return chosen[:n]


# ════════════════════════════════════════════════════════════════════════════
# DEEPSEEK OPTIMIZATION + SUPABASE PUSH
# ════════════════════════════════════════════════════════════════════════════

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
                    {"role": "user",   "content": title},
                ],
            )
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("  optimise_title error:", e)
        return title


async def ai_intelligence(raw_title: str, state: str = ""):
    print(f"  🤖 AI processing: {raw_title[:70]}")
    optimized = await optimise_title(raw_title)
    location_context = f"\nLocation Context: {state}" if state else ""
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
                    {"role": "user",   "content": prompt},
                ],
            )
        )
        output = res.choices[0].message.content.strip()

        if output.strip().upper() == "SKIP":
            print(f"  ⏭️  Skipped: {raw_title[:70]}")
            return

        supabase.table(TABLE_NAME).insert({
            "tittle":         optimized,
            "regular_tittle": raw_title,
            "summary":        output,
            "state":          state,
        }).execute()
        print(f"  ✅ Saved → {TABLE_NAME} [{state}]: {optimized}")
    except Exception as e:
        print(f"  AI error:", e)


async def _process_bucket(titles: list[str], state: str, semaphore: asyncio.Semaphore):
    """Run ai_intelligence for every title in a bucket, bounded by semaphore."""
    async def _run(t):
        async with semaphore:
            await ai_intelligence(t, state)

    await asyncio.gather(*(_run(t) for t in titles))


async def push_to_db(
    national_csv: Path = OUTPUT_DIR / "national_titles.csv",
    regional_csv: Path = OUTPUT_DIR / "regional_titles.csv",
    global_csv: Path = OUTPUT_DIR / "global_titles.csv",
    per_bucket: int = PER_BUCKET_TARGET,
):
    """
    Optimise + summarise titles via DeepSeek and push to Supabase `content_radar`.
      - National  → exactly `per_bucket` rows, state = "national"
      - Global    → exactly `per_bucket` rows, state = "global"
      - Regional  → exactly `per_bucket` rows PER STATE, state = state name
    """
    semaphore = asyncio.Semaphore(AI_CONCURRENCY)

    # ── National ─────────────────────────────────────────────────────────────
    if national_csv.exists():
        df_nat = pd.read_csv(national_csv)
        nat_titles = pick_top_n_titles(df_nat, n=per_bucket)
        print(f"\n📰 National → pushing {len(nat_titles)} titles (target {per_bucket})")
        await _process_bucket(nat_titles, "national", semaphore)
    else:
        print(f"\n⚠️  {national_csv} not found — skipping national push")

    # ── Global ───────────────────────────────────────────────────────────────
    if global_csv.exists():
        df_glob = pd.read_csv(global_csv)
        # global_titles.csv uses "country" column instead of "region"; normalize
        if "source" not in df_glob.columns and "country" in df_glob.columns:
            pass  # source column already present from collection step
        glob_titles = pick_top_n_titles(df_glob, n=per_bucket)
        print(f"\n🌍 Global → pushing {len(glob_titles)} titles (target {per_bucket})")
        await _process_bucket(glob_titles, "global", semaphore)
    else:
        print(f"\n⚠️  {global_csv} not found — skipping global push")

    # ── Regional (per state) ────────────────────────────────────────────────
    if regional_csv.exists():
        df_reg = pd.read_csv(regional_csv)
        if "region" in df_reg.columns:
            states = sorted(df_reg["region"].dropna().unique().tolist())
            for state in states:
                df_state = df_reg[df_reg["region"] == state]
                state_titles = pick_top_n_titles(df_state, n=per_bucket)
                print(f"\n🗺️  {state} → pushing {len(state_titles)} titles (target {per_bucket})")
                await _process_bucket(state_titles, state, semaphore)
        else:
            print(f"\n⚠️  {regional_csv} missing 'region' column — skipping regional push")
    else:
        print(f"\n⚠️  {regional_csv} not found — skipping regional push")

    print("\n🎉 DB push complete!")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run():
    try:
        df_regional = pd.read_csv("./Regional.csv")
    except FileNotFoundError:
        print("⚠️  Regional.csv not found — using empty DataFrame")
        df_regional = pd.DataFrame(columns=["State", "Source", "URL", "Selector"])

    try:
        df_national = pd.read_csv("./National.csv")
    except FileNotFoundError:
        print("⚠️  National.csv not found — using empty DataFrame")
        df_national = pd.DataFrame(columns=["Source Name", "url", "Selector"])

    national_out = OUTPUT_DIR / "national_titles.csv"
    if national_out.exists():
        print(f"\n📰 Skipping National fetch — {national_out} already exists")
    else:
        rows = get_national_news(df_national)
        if rows:
            pd.DataFrame(rows).to_csv(national_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(rows)} national titles saved → {national_out}")
        else:
            print("\n⚠️  No national titles saved")

    regional_out = OUTPUT_DIR / "regional_titles.csv"
    if regional_out.exists():
        print(f"\n🗺️  Skipping Regional fetch — {regional_out} already exists")
    else:
        rows = get_regional_news(df_regional)
        if rows:
            pd.DataFrame(rows).to_csv(regional_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(rows)} regional titles saved → {regional_out}")
        else:
            print("\n⚠️  No regional titles collected")

    global_out = OUTPUT_DIR / "global_titles.csv"
    if global_out.exists():
        print(f"\n🌍 Skipping Global fetch — {global_out} already exists")
    else:
        rows = get_global_news()
        if rows:
            pd.DataFrame(rows).to_csv(global_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(rows)} global titles saved → {global_out}")
        else:
            print("\n⚠️  No global titles collected")

    find_common_news(national_csv=national_out, threshold=0.45, top_n=10)

    # ── Push optimized titles + summaries to Supabase ─────────────────────────
    asyncio.run(push_to_db(
        national_csv=national_out,
        regional_csv=regional_out,
        global_csv=global_out,
        per_bucket=PER_BUCKET_TARGET,
    ))

    print("\n🎉 Done!")


if __name__ == "__main__":
    run()