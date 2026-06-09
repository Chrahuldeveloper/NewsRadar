import os
import time
import asyncio
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GNEWS_KEY    = os.getenv("GnewsApi", "").strip()
NEWSDATA_KEY = os.getenv("Newsdata_api_key", "").strip()

OUTPUT_DIR = Path("news_output")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

COUNTRY_CODES = {
    "United States": "us", "Canada": "ca", "Mexico": "mx",
    "United Kingdom": "gb", "Germany": "de", "France": "fr",
    "Australia": "au", "China": "cn", "Japan": "jp",
    "India": "in", "Brazil": "br", "South Africa": "za",
    "Russia": "ru", "Italy": "it", "Spain": "es",
    "Netherlands": "nl", "Poland": "pl", "South Korea": "kr",
    "Taiwan": "tw", "Pakistan": "pk", "Bangladesh": "bd",
    "Sri Lanka": "lk", "Saudi Arabia": "sa", "UAE": "ae",
    "Israel": "il", "Iran": "ir", "Turkey": "tr",
    "Nigeria": "ng", "Egypt": "eg", "Kenya": "ke",
    "Ethiopia": "et", "New Zealand": "nz",
}

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ── AI ────────────────────────────────────────────────────────────────────────

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
        output = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You generate viral content."},
                    {"role": "user",   "content": prompt},
                ],
            ).choices[0].message.content.strip()
        )

        if output.strip().upper() == "SKIP":
            print(f"  ⏭️  Skipped: {raw_title[:70]}")
            return

        supabase.table("content_radar").insert({
            "tittle":         optimized,
            "regular_tittle": raw_title,
            "summary":        output,
            "state":          state,
        }).execute()

        print(f"  ✅ Saved → content_radar [{state}]: {optimized}")

    except Exception as e:
        print(f"  AI error:", e)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _safe_get(url, params=None, tag=""):
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
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


# ── SCRAPER ───────────────────────────────────────────────────────────────────

def scrape_titles(url, selector=None):
    resp, err = _safe_get(url, tag="SCRAPE")
    if resp is None:
        return []

    soup   = BeautifulSoup(resp.content, "html.parser")
    titles = []

    if "sakshi.com" in url:
        for el in soup.select("h2, h3, a, .title, .story-title, .news-title"):
            text = el.get_text(strip=True)
            if text and len(text) > 25:
                titles.append(text)
        return list(dict.fromkeys(titles))

    if selector and selector.lower() not in ("", "nan", "none"):
        for el in soup.select(selector):
            text = el.get_text(strip=True)
            if text:
                titles.append(text)
        return list(dict.fromkeys(titles))

    for el in soup.select("h1, h2, h3"):
        text = el.get_text(strip=True)
        if text:
            titles.append(text)

    return list(dict.fromkeys(titles))


def get_regional_titles(url, selector):
    print("   → scraping ...", end=" ", flush=True)
    titles = scrape_titles(url, selector)
    print(f"{len(titles)} titles" if titles else "⚠️  none found")
    return titles


# ── GNEWS ─────────────────────────────────────────────────────────────────────

def _fetch_gnews(country_code, tag="GNews"):
    if not GNEWS_KEY:
        print(f"   [{tag}] ⚠️  GnewsApi env var not set — skipping")
        return []
    resp, _ = _safe_get(
        "https://gnews.io/api/v4/top-headlines",
        params={"country": country_code, "max": 10, "apikey": GNEWS_KEY},
        tag=tag,
    )
    if resp is None:
        return []
    data     = resp.json()
    articles = data.get("articles", [])
    if not articles:
        print(f"   [{tag}] ⚠️  Response OK but 0 articles. Raw: {str(data)[:200]}")
    return [a.get("title", "").strip() for a in articles if a.get("title")]


def _fetch_newsdata(country_code, tag="NewsData"):
    if not NEWSDATA_KEY:
        print(f"   [{tag}] ⚠️  Newsdata_api_key env var not set — skipping")
        return []
    resp, _ = _safe_get(
        "https://newsdata.io/api/1/news",
        params={"apikey": NEWSDATA_KEY, "country": country_code, "language": "en"},
        tag=tag,
    )
    if resp is None:
        return []
    data = resp.json()
    if data.get("status") != "success":
        print(f"   [{tag}] ❌ API status={data.get('status')} — {str(data)[:200]}")
        return []
    results = data.get("results") or []
    if not results:
        print(f"   [{tag}] ⚠️  Response OK but 0 results.")
    return [a.get("title", "").strip() for a in results if a.get("title")]


def _fetch_news(country_code, label=""):
    titles = _fetch_gnews(country_code, tag=f"GNews/{label}")
    if titles:
        return titles
    print(f"   GNews returned nothing — trying NewsData ...")
    titles = _fetch_newsdata(country_code, tag=f"NewsData/{label}")
    return titles


def get_global_news(country, country_code):
    print(f"  🌐 {country} ({country_code}) ...", end=" ", flush=True)
    titles = _fetch_news(country_code, label=country)
    print(f"✅ {len(titles)}" if titles else "❌ none")
    return titles


# ── NATIONAL NEWS ─────────────────────────────────────────────────────────────

def get_national_news(df_national):
    print("\n📰 Fetching National News (scrape-based)...")
    print("─" * 50)

    national_rows = []
    for _, row in df_national.iterrows():
        source   = str(row.get("Source Name", "")).strip()
        url      = str(row.get("url",         "")).strip()
        selector = str(row.get("Selector",    "")).strip()

        if not url or url.lower() in ("nan", "none", ""):
            print(f"\n  ⚠️  {source} — no URL, skipping")
            continue

        print(f"\n  📌 National → {source}")
        print("  " + "─" * 40)
        print("   → scraping ...", end=" ", flush=True)

        titles = scrape_titles(url, selector)
        print(f"{len(titles)} titles" if titles else "⚠️  none found")

        for title in titles:
            national_rows.append({"region": "National", "source": source, "title": title})

        time.sleep(1)

    return national_rows


# ── TRANSLATION (REGIONAL ONLY) ───────────────────────────────────────────────

def translate_titles_to_english(df, label=""):
    try:
        from deep_translator import GoogleTranslator
        from langdetect import detect
    except ImportError:
        print("   ⚠️  Translation libs not installed — using original titles.")
        print("      Run:  pip install deep-translator langdetect")
        df = df.copy()
        df["title_en"] = df["title"]
        return df

    translated = []
    translator = GoogleTranslator(source="auto", target="en")
    total      = len(df)
    print(f"   🌐 [{label}] Translating {total} titles to English...")

    for i, title in enumerate(df["title"]):
        try:
            from langdetect import detect
            lang = detect(title)
        except Exception:
            lang = "en"

        if lang == "en":
            translated.append(title)
        else:
            try:
                result = translator.translate(title)
                translated.append(result if result else title)
            except Exception:
                translated.append(title)

        if (i + 1) % 100 == 0:
            print(f"      ... {i + 1}/{total} done")

    df = df.copy()
    df["title_en"] = translated
    print(f"   ✅ [{label}] Translation done")
    return df


# ── CLUSTERING ────────────────────────────────────────────────────────────────

def _cluster_titles(df, model, threshold):
    from sentence_transformers import util

    titles = df["title_en"].tolist()
    if not titles:
        return []

    embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    cos_sim    = util.cos_sim(embeddings, embeddings)
    n          = len(titles)
    assigned   = [False] * n
    clusters   = []

    connectivity = [(int((cos_sim[i] >= threshold).sum()) - 1, i) for i in range(n)]
    connectivity.sort(reverse=True)

    for _, i in connectivity:
        if assigned[i]:
            continue

        similar_indices = [
            j for j in range(n)
            if not assigned[j] and i != j and float(cos_sim[i][j]) >= threshold
        ]

        if not similar_indices:
            continue

        cluster_indices = [i] + similar_indices
        for idx in cluster_indices:
            assigned[idx] = True

        sources     = df.iloc[cluster_indices]["source"].tolist()
        orig_titles = df.iloc[cluster_indices]["title"].tolist()
        en_titles   = df.iloc[cluster_indices]["title_en"].tolist()
        unique_sources = list(dict.fromkeys(sources))

        clusters.append({
            "canonical_title_en":   en_titles[0],
            "canonical_title_orig": orig_titles[0],
            "sources":              " | ".join(unique_sources),
            "unique_sources":       len(unique_sources),
            "repeat_count":         len(cluster_indices),
            "matched_titles_en":    " ||| ".join(en_titles),
            "matched_titles_orig":  " ||| ".join(orig_titles),
        })

    return sorted(clusters, key=lambda x: (x["unique_sources"], x["repeat_count"]), reverse=True)


def _print_top(clusters, label, top_n):
    print(f"\n   🔥 Top {top_n} Hot Topics [{label}]:")
    print("   " + "─" * 60)
    for rank, c in enumerate(clusters[:top_n], 1):
        print(f"\n   #{rank} [{c['repeat_count']}x | {c['unique_sources']} sources]")
        print(f"   {c['canonical_title_en'][:90]}")
        print(f"   📰 {c['sources'][:120]}")


# ── FIND TOP REPEATED + SAVE TO DB ───────────────────────────────────────────

async def find_semantic_matches(national_csv, regional_csv, threshold=0.5, top_n=10):
    print("\n🔍 Running Semantic Similarity Search...")
    print("─" * 50)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("   ❌ sentence-transformers not installed.")
        print("      Run:  pip install sentence-transformers")
        return

    def _load(path, dataset_label, keep_region=False):
        if not Path(path).exists():
            print(f"   ⚠️  {path} not found — skipping")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        if "source name" in df.columns:
            df = df.rename(columns={"source name": "source"})
        cols = ["source", "title"] + (["region"] if keep_region and "region" in df.columns else [])
        df = df[cols].dropna(subset=["title"])
        df["title"]   = df["title"].astype(str).str.strip()
        df["source"]  = df["source"].astype(str).str.strip()
        df["dataset"] = dataset_label
        return df.reset_index(drop=True)

    df_nat = _load(national_csv, "national")
    df_reg = _load(regional_csv, "regional", keep_region=True)

    if df_nat.empty and df_reg.empty:
        print("   ❌ Both CSVs are empty — nothing to compare.")
        return

    # national — no translation
    if not df_nat.empty:
        print(f"\n   📰 National: {len(df_nat)} titles (no translation needed)")
        df_nat["title_en"] = df_nat["title"]

    # regional — translate
    if not df_reg.empty:
        print(f"\n   🗺️  Regional: {len(df_reg)} titles")
        df_reg = translate_titles_to_english(df_reg, label="Regional")

    print("\n   Loading model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── 1. NATIONAL ───────────────────────────────────────────────
    if not df_nat.empty:
        print(f"\n   [1/3] Clustering NATIONAL titles ({len(df_nat)}) ...")
        nat_clusters = _cluster_titles(df_nat, model, threshold)
        _print_top(nat_clusters, "National", top_n)
        print(f"\n   💾 Sending top {top_n} National → AI + DB")
        for c in nat_clusters[:top_n]:
            await ai_intelligence(c["canonical_title_en"], state="National")

    # ── 2. REGIONAL (per state) ───────────────────────────────────
    if not df_reg.empty:
        print(f"\n   [2/3] Clustering REGIONAL titles per state ...")
        for region_name, group in df_reg.groupby("region", sort=False):
            group = group.reset_index(drop=True)
            print(f"\n   📌 {region_name} ({len(group)} titles)")
            reg_clusters = _cluster_titles(group, model, threshold)
            _print_top(reg_clusters, region_name, top_n)
            print(f"\n   💾 Sending top {top_n} [{region_name}] → AI + DB")
            for c in reg_clusters[:top_n]:
                await ai_intelligence(c["canonical_title_en"], state=region_name)

    # ── 3. GLOBAL ─────────────────────────────────────────────────
    global_csv = OUTPUT_DIR / "global_titles.csv"
    if global_csv.exists():
        df_glob = pd.read_csv(global_csv)
        df_glob.columns = df_glob.columns.str.strip().str.lower()
        df_glob = df_glob.rename(columns={"country": "source"})
        df_glob = df_glob[["source", "title"]].dropna(subset=["title"])
        df_glob["title"]    = df_glob["title"].astype(str).str.strip()
        df_glob["source"]   = df_glob["source"].astype(str).str.strip()
        df_glob["title_en"] = df_glob["title"]
        df_glob = df_glob.reset_index(drop=True)

        print(f"\n   [3/3] Clustering GLOBAL titles ({len(df_glob)}) ...")
        glob_clusters = _cluster_titles(df_glob, model, threshold)
        _print_top(glob_clusters, "Global", top_n)
        print(f"\n   💾 Sending top {top_n} Global → AI + DB")
        for c in glob_clusters[:top_n]:
            await ai_intelligence(c["canonical_title_en"], state="Global")
    else:
        print("\n   ⚠️  global_titles.csv not found — skipping global clustering")


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def run():
    try:
        df_regional = pd.read_csv("./Regional.csv")
    except FileNotFoundError:
        print("❌ Regional.csv not found — skipping regional section")
        df_regional = pd.DataFrame(columns=["State", "Source", "URL", "Selector"])

    try:
        df_global = pd.read_csv("./Global.csv")
    except FileNotFoundError:
        print("❌ Global.csv not found — skipping global section")
        df_global = pd.DataFrame(columns=["Country", "Region"])

    try:
        df_national = pd.read_csv("./National.csv")
    except FileNotFoundError:
        print("❌ National.csv not found — skipping national section")
        df_national = pd.DataFrame(columns=["Source Name", "Source Type", "Primary Language", "Coverage", "url"])

    # ── GLOBAL ────────────────────────────────────────────────────
    global_out = OUTPUT_DIR / "global_titles.csv"
    if global_out.exists():
        print(f"\n🌍 Skipping Global fetch — {global_out} already exists")
    else:
        print("\n🌍 Fetching Global News...")
        print("─" * 50)
        global_rows = []
        for _, row in df_global.iterrows():
            country = str(row.get("Country", "")).strip()
            region  = str(row.get("Region",  "")).strip()
            code = COUNTRY_CODES.get(country)
            if not code:
                print(f"   ⚠️  No country code mapped for '{country}' — skipping")
                continue
            for title in get_global_news(country, code):
                global_rows.append({"geo_region": region, "country": country, "title": title})
            time.sleep(1)

        if global_rows:
            pd.DataFrame(global_rows).to_csv(global_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(global_rows)} global titles saved → {global_out}")
        else:
            print("\n⚠️  No global titles collected")

    # ── NATIONAL ──────────────────────────────────────────────────
    national_out = OUTPUT_DIR / "national_titles.csv"
    if national_out.exists():
        print(f"\n📰 Skipping National fetch — {national_out} already exists")
    else:
        national_rows = get_national_news(df_national)
        if national_rows:
            pd.DataFrame(national_rows).to_csv(national_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(national_rows)} national titles saved → {national_out}")
        else:
            print("\n⚠️  No national titles saved")

    # ── REGIONAL ──────────────────────────────────────────────────
    regional_out = OUTPUT_DIR / "regional_titles.csv"
    if regional_out.exists():
        print(f"\n🗺️  Skipping Regional fetch — {regional_out} already exists")
    else:
        print("\n🗺️  Fetching Regional News (CSV-driven)...")
        print("─" * 50)
        regional_rows = []
        for _, row in df_regional.iterrows():
            state    = str(row.get("State",    "")).strip()
            source   = str(row.get("Source",   "")).strip()
            url      = str(row.get("URL",      "")).strip()
            selector = str(row.get("Selector", "")).strip()

            if not url or url.lower() in ("nan", "none", ""):
                print(f"\n  ⚠️  {state} → {source} — no URL, skipping")
                continue

            print(f"\n  📌 {state} → {source}")
            print("  " + "─" * 40)
            for title in get_regional_titles(url, selector):
                regional_rows.append({"region": state, "source": source, "title": title})
            time.sleep(1)

        if regional_rows:
            pd.DataFrame(regional_rows).to_csv(regional_out, index=False, encoding="utf-8-sig")
            print(f"\n✅ {len(regional_rows)} regional titles saved → {regional_out}")
        else:
            print("\n⚠️  No regional titles collected")

    # ── CLUSTER + AI + SAVE TO DB ─────────────────────────────────
    await find_semantic_matches(
        national_csv = national_out,
        regional_csv = regional_out,
        threshold    = 0.5,
        top_n        = 10,
    )

    print("\n🎉 Done!")


if __name__ == "__main__":
    asyncio.run(run())