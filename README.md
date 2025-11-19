# PL Reddit Sentiment + Odds (SQLite)

Collect Premier League match discussion from Reddit, score every comment with VADER sentiment + MiniLM embeddings, join that chatter to historical bookmaker odds, and explore everything in a lightweight Streamlit dashboard backed by SQLite.

## Highlights
- **One-file database** – all threads, comments, odds, and sentiment live in `pl.db`.
- **No API keys** – scrapes Reddit’s old UI HTML and Football-Data CSVs.
- **Embeddings + sentiment** – VADER scores and sentence-transformer vectors stored for later experiments.
- **Streamlit UI** – inspect keywords, top threads, latest odds, and pre-match sentiment vs implied probabilities.
- **Model comparison CLI** – quantify how bookmaker odds, sentiment-only, or combined signals perform via logistic CV.

---

## Project Layout
- `app.py` – Streamlit dashboard (takes `--db`).
- `cli.py` – Typer command surface for ingest + analysis.
- `db.py` – SQLite helpers and schema (threads, comments, matches).
- `scrape_reddit.py` – HTML scraper for `r/soccer` + `r/PremierLeague`.
- `sentiment_embed.py` – VADER + `sentence-transformers/all-MiniLM-L6-v2`.
- `odds_loader.py` – pulls Bet365 odds from football-data.co.uk CSVs.
- `pl.db` – example database (WAL files alongside).

---

## Prerequisites
- Python 3.10+ recommended.
- `pip` (or `uv`, `pip-tools`, etc.).
- macOS/Linux: `bash`, `curl`. Windows: use WSL2 or PowerShell.
- Internet access for Reddit HTML, Football-Data CSV, NLTK lexicon, SentenceTransformer weights.

---

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

NLTK downloads the VADER lexicon automatically on first run. SentenceTransformer will cache its weights in `~/.cache/torch/sentence_transformers`.

---

## End-to-End Pipeline
Each step writes into the same SQLite file (default `pl.db`). Provide `--db path/to/file.db` if you want multiple runs.

| Step | Command | Purpose |
| --- | --- | --- |
| 1. Initialize schema | `python cli.py init-db --db pl.db` | Creates `threads`, `comments`, `matches`. |
| 2. Scrape Reddit | `python cli.py scrape "Arsenal,Tottenham,Liverpool" --db pl.db` | Pulls recent posts from `r/soccer` & `r/PremierLeague`, stores threads + raw comments per keyword. |
| 3. Load odds | `python cli.py load-odds 2324 --db pl.db` | Downloads season CSV (format `YYZZ` e.g. `2324`) and inserts Bet365 odds + implied probs. |
| 4. Link threads to matches | `python cli.py link-threads --db pl.db` | Detects team names in thread titles, finds nearest scheduled match, fills `match_id`. |
| 5. Score sentiment + embeddings | `python cli.py analyze --db pl.db --limit 5000` | Runs VADER + MiniLM on unanalyzed comments (batched). Repeat until queue empty or adjust `--limit`. |
| 6. Explore UI | `streamlit run app.py -- --db pl.db` | Metrics, keyword averages, top positive threads, odds table, sentiment vs odds. |
| 7. (Optional) Compare predictors | `python cli.py compare-predictors --db pl.db --min-comments 15` | Builds match-level features and cross-validates odds, sentiment, and combined models. |

Tips:
- Adjust scrape keywords to cover both team nicknames and full names. Update `HEADERS` in `scrape_reddit.py` with your own User-Agent to be polite.
- Rerun `scrape` regularly; duplicates are ignored via `INSERT OR IGNORE`.
- If linking misses fixtures, lower the window in `_nearest_match` or add aliases in `cli.py`.

---

## Running the Streamlit App
```bash
streamlit run app.py -- --db pl.db
```

The Streamlit script parses `--db` itself (everything after `--`). Sections unlock progressively:
- Keyword-level sentiment summary + bar chart (after `comments` populated + analyzed).
- Top positive threads with ≥10 comments.
- Latest matches table once odds are loaded.
- Linked thread → match sentiment scatter when `link-threads` + `analyze` have run.

Deploy Streamlit Cloud or share by running `streamlit run` and forwarding port 8501.

---

## Data Model
```
threads(id, keyword, sub, title, url, created_utc, n_comments, match_id*)
comments(id, thread_url, keyword, sub, author, body, created_utc, sentiment, emb_dim, emb_json)
matches(id, season, date, home, away, result, odds_h/d/a, p_h/d/a)
```
Indexes on `comments.thread_url`, `comments.keyword`, `matches.date` keep lookups snappy. Comments store embeddings as JSON + dimension field for future ML work.

---

## Troubleshooting
- **Rate limiting / empty scrapes** – Reddit HTML endpoints throttle. Increase `time.sleep`, rotate keywords, or add more subreddits.
- **SentenceTransformer download issues** – pre-download with `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"` inside the venv.
- **Not enough sentiment samples** – lower `--min-comments` for `compare-predictors` or gather more data.
- **Streamlit empty states** – sections show info messages until prerequisite tables have rows.

---

## Extending
- Add more subs or seasons by editing constants in `scrape_reddit.py` / `TEAMS` in `cli.py`.
- Store additional bookmaker columns in `matches`.
- Swap out sentiment model in `sentiment_embed.py` (be mindful of embedding size stored in SQLite).
- Schedule scrapes with cron/GitHub Actions; SQLite WAL mode already enabled in `db.py`.

Happy tinkering! File issues or PRs if you add new predictors, visualizations, or data sources.
