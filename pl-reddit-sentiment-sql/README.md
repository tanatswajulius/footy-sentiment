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
- Python 3.10+.
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
| 8. (Optional) Team profiles | `python cli.py team-profiles --db pl.db --output-csv profiles.csv` | Aggregates sentiment per team, correlates with historical form, exports CSV. |

Tips:
- Adjust scrape keywords to cover both team nicknames and full names. Update `HEADERS` in `scrape_reddit.py` with your own User-Agent to be polite.
- Rerun `scrape` regularly; duplicates are ignored via `INSERT OR IGNORE`.
- If linking misses fixtures, lower the window in `_nearest_match` or add aliases in `cli.py`.

---

## Running the Streamlit App
```bash
streamlit run app.py -- --db pl.db
```

The Streamlit script parses `--db` itself (everything after `--`). 

### Dashboard Pages

| Page | What It Shows |
|------|---------------|
| **Overview** | Metrics, sentiment by keyword bar chart, top positive threads |
| **Team Analysis** | Sentiment rankings, box plots, scatter (sentiment vs win%), outcome breakdown |
| **Matches & Odds** | Season filter, match table with implied probabilities, outcome distribution |
| **Raw Data** | Browse threads/comments/matches tables directly |

Sections unlock progressively as you populate data via CLI commands.

---

## Deploying to Streamlit Cloud

The app comes with a pre-populated `pl.db` database (48MB) containing threads, comments, and match odds ready for exploration.

### Quick Deploy Steps

1. **Push to GitHub** (if not already):
   ```bash
   git add pl.db app.py cli.py db.py requirements.txt .streamlit/
   git commit -m "Add static data for deployment"
   git push
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Set **Main file path**: `pl-reddit-sentiment-sql/app.py`
   - Click "Deploy"

3. **That's it!** The app will use the bundled `pl.db` automatically.

### Notes
- The database is read-only on Streamlit Cloud (no new scrapes/analysis in prod)
- To update data: run locally, re-commit `pl.db`, redeploy
- For larger databases (>100MB), use Git LFS or external storage

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
