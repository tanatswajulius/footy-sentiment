import time, requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import List, Dict

HEADERS = {"User-Agent": "pl-reddit-sentiment-sql/0.1 by yourname"}
SUBS = ["soccer", "PremierLeague"]

def search_posts(query: str, per_sub_limit: int = 30) -> List[Dict]:
    rows = []
    for sub in SUBS:
        url = f"https://old.reddit.com/r/{sub}/search/?q={quote_plus(query)}&restrict_sr=1&sort=new&t=week"
        html = requests.get(url, headers=HEADERS, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        count = 0
        for a in soup.select("a.search-title"):
            href = a.get("href")
            title = a.text.strip()
            if not href:
                continue
            rows.append({"sub": sub, "title": title, "url": href})
            count += 1
            if count >= per_sub_limit:
                break
        time.sleep(0.7)
    return rows

def search_posts_pushshift(
    query: str,
    after_ts: int,
    before_ts: int,
    per_sub_limit: int = 80,
    subs: List[str] = None,
) -> List[Dict]:
    """Discover submissions via Pushshift within a date window, newest first."""
    subs = subs or SUBS
    rows: List[Dict] = []
    bases = [
        "https://api.pushshift.io/reddit/search/submission/",
        "https://api.pullpush.io/reddit/search/submission/",
    ]
    for sub in subs:
        data = []
        params = {
                "q": query,
                "after": after_ts,
                "before": before_ts,
                "subreddit": sub,
                "size": per_sub_limit,
                "sort": "desc",
                "sort_type": "created_utc",
            }
        for base in bases:
            try:
                r = requests.get(base, params=params, headers=HEADERS, timeout=20)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    break
            except Exception:
                continue
        for d in data:
            href = d.get("url") or d.get("full_link")
            title = d.get("title")
            created = d.get("created_utc")
            if not href or not title:
                continue
            rows.append({"sub": sub, "title": title.strip(), "url": href, "created_utc": created})
        time.sleep(0.3)
    return rows

def fetch_submission_and_comments(post_url: str, max_comments: int = 300):
    if not post_url.endswith(".json"):
        post_url = post_url.rstrip("/") + ".json"
    r = requests.get(post_url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return None, []
    try:
        js = r.json()
    except Exception:
        return None, []

    created_utc = None
    try:
        created_utc = js[0]["data"]["children"][0]["data"].get("created_utc")
    except Exception:
        pass

    comments = []
    try:
        for obj in js[1]["data"]["children"]:
            data = obj.get("data", {})
            body = data.get("body","" )
            if body and body not in ("[deleted]","[removed]"):
                comments.append({
                    "author": data.get("author"),
                    "body": body,
                    "created_utc": data.get("created_utc")
                })
                if len(comments) >= max_comments:
                    break
    except Exception:
        pass
    return created_utc, comments
