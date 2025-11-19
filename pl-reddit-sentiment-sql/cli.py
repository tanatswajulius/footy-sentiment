from __future__ import annotations
import re, json, datetime as dt
from typing import List, Optional
import typer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss
from db import init_schema, connect
from scrape_reddit import search_posts, fetch_submission_and_comments
from odds_loader import load_football_data_csv
from sentiment_embed import Analyzer

app = typer.Typer(add_completion=False)

TEAMS = [
  "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton",
  "Chelsea","Crystal Palace","Everton","Fulham","Ipswich","Leicester",
  "Liverpool","Man City","Man United","Newcastle","Nottingham Forest",
  "Southampton","Tottenham","West Ham","Wolves"
]
ALIASES = {
  "Manchester City":"Man City", "Man City":"Man City", "MCFC":"Man City", "City":"Man City",
  "Manchester United":"Man United", "Man Utd":"Man United", "Man United":"Man United", "United":"Man United",
  "Spurs":"Tottenham", "Tottenham Hotspur":"Tottenham",
  "Nottingham":"Nottingham Forest", "Nottm Forest":"Nottingham Forest",
  "Wolverhampton":"Wolves", "Wolves":"Wolves",
  "Brighton & Hove Albion":"Brighton", "Brighton and Hove Albion":"Brighton",
  "West Ham United":"West Ham", "Crystal P":"Crystal Palace",
}

def canonicalize(token: str) -> Optional[str]:
    token = ALIASES.get(token, token)
    for t in TEAMS:
        if t.lower() in token.lower():
            return t
    return None

def extract_teams_from_title(title: str) -> List[str]:
    found = set()
    for t in TEAMS + list(ALIASES.keys()):
        if re.search(rf"\b{re.escape(t)}\b", title, flags=re.I):
            cand = canonicalize(t)
            if cand: found.add(cand)
    if "Man City" in title and "Man United" in title:
        found.update({"Man City","Man United"})
    return list(found)[:2]

@app.command()
def init_db(db: str = typer.Option("pl.db")):
    init_schema(db)
    typer.echo(f"Initialized schema at {db}")

@app.command()
def scrape(keywords: str, db: str = typer.Option("pl.db")):
    kw_list: List[str] = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        typer.echo("No keywords provided."); raise typer.Exit(1)
    with connect(db) as conn:
        for kw in kw_list:
            posts = search_posts(kw, per_sub_limit=30)
            for p in posts:
                created_utc, comments = fetch_submission_and_comments(p["url"], max_comments=300)
                conn.execute(
                    "INSERT OR IGNORE INTO threads(keyword, sub, title, url, created_utc, n_comments) VALUES(?,?,?,?,?,?)",
                    (kw, p["sub"], p["title"], p["url"], float(created_utc) if created_utc else None, int(len(comments)))
                )
                for c in comments:
                    conn.execute(
                        """INSERT INTO comments(thread_url, keyword, sub, author, body, created_utc)
                           VALUES(?,?,?,?,?,?)""",
                        (p["url"], kw, p["sub"], c.get("author"), c.get("body"), float(c.get("created_utc")) if c.get("created_utc") else None)
                    )
    typer.echo("Scrape complete.")

@app.command()
def load_odds(season: str, db: str = typer.Option("pl.db")):
    df = load_football_data_csv(season)
    with connect(db) as conn:
        for _, r in df.iterrows():
            conn.execute(
                """INSERT OR IGNORE INTO matches(season,date,home,away,result,odds_h,odds_d,odds_a,p_h,p_d,p_a)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (season, r["date"], r["home"], r["away"], r["result"],
                 float(r["odds_h"]), float(r["odds_d"]), float(r["odds_a"]),
                 float(r["p_h"]), float(r["p_d"]), float(r["p_a"]))
            )
    typer.echo(f"Loaded {len(df)} matches for season {season}.")

def _nearest_match(conn, team_a: str, team_b: str, ts_utc: float) -> Optional[int]:
    if ts_utc:
        thread_date = dt.datetime.utcfromtimestamp(ts_utc).date()
        for window in (3,7):
            start = (thread_date - dt.timedelta(days=window)).isoformat()
            end   = (thread_date + dt.timedelta(days=window)).isoformat()
            q = """SELECT id, date FROM matches
                   WHERE date BETWEEN ? AND ? AND
                         ((home=? AND away=?) OR (home=? AND away=?))"""
            rows = conn.execute(q, (start,end,team_a,team_b,team_b,team_a)).fetchall()
            if rows:
                best_id, best_delta = None, 9999
                for mid, dstr in rows:
                    d = dt.date.fromisoformat(dstr)
                    delta = abs((d - thread_date).days)
                    if delta < best_delta: best_id, best_delta = mid, delta
                return best_id
    row = conn.execute("""SELECT id FROM matches
                          WHERE (home=? AND away=?) OR (home=? AND away=?)
                          ORDER BY date DESC LIMIT 1""", (team_a,team_b,team_b,team_a)).fetchone()
    return row[0] if row else None

@app.command("link-threads")
def link_threads(db: str = typer.Option("pl.db")):
    with connect(db) as conn:
        rows = conn.execute("SELECT id, title, created_utc FROM threads WHERE match_id IS NULL").fetchall()
        linked = 0
        for tid, title, created_utc in rows:
            teams = extract_teams_from_title(title or "")
            if len(teams) < 2: continue
            m_id = _nearest_match(conn, teams[0], teams[1], created_utc or 0.0)
            if m_id:
                conn.execute("UPDATE threads SET match_id=? WHERE id=?", (m_id, tid))
                linked += 1
        typer.echo(f"Linked {linked} thread(s) to matches.")

@app.command()
def analyze(db: str = typer.Option("pl.db"), limit: int = typer.Option(5000)):
    an = Analyzer()
    with connect(db) as conn:
        df = pd.read_sql_query("SELECT id, body FROM comments WHERE sentiment IS NULL LIMIT ?", conn, params=(int(limit),))
        if df.empty:
            typer.echo("No unanalyzed comments found."); raise typer.Exit()
        texts = df["body"].fillna("").tolist()
        sents = [an.score_sentiment(t) for t in texts]
        vecs, dim = an.embed(texts)
        for cid, sent, vec in zip(df["id"].tolist(), sents, vecs):
            conn.execute("UPDATE comments SET sentiment=?, emb_dim=?, emb_json=? WHERE id=?",
                         (float(sent), int(dim), json.dumps(vec), int(cid)))
    typer.echo(f"Analyzed {len(df)} comments.")

def _build_sentiment_feature_frame(conn, min_comments: int) -> pd.DataFrame:
    matches = pd.read_sql_query(
        "SELECT id, season, date, home, away, result, p_h, p_d, p_a FROM matches",
        conn
    )
    if matches.empty:
        return pd.DataFrame()
    matches["date_dt"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date_dt"])
    comments = pd.read_sql_query(
        """SELECT t.match_id, c.keyword, c.sentiment, c.created_utc AS comment_utc
           FROM threads t
           JOIN comments c ON c.thread_url = t.url
           WHERE t.match_id IS NOT NULL AND c.sentiment IS NOT NULL""",
        conn
    )
    if comments.empty:
        return pd.DataFrame()
    comments["comment_dt"] = pd.to_datetime(comments["comment_utc"], unit="s", errors="coerce")
    comments = comments.dropna(subset=["comment_dt"])
    match_dates = matches.set_index("id")["date_dt"]
    comments["match_date"] = comments["match_id"].map(match_dates)
    comments = comments.dropna(subset=["match_date"])
    comments = comments[comments["comment_dt"] <= comments["match_date"] + pd.Timedelta(hours=12)]
    comments["team"] = comments["keyword"].apply(lambda x: canonicalize(x) if isinstance(x, str) else None)
    comments = comments.dropna(subset=["team"])
    if comments.empty:
        return pd.DataFrame()
    agg = (comments
           .groupby(["match_id","team"])
           .agg(sent_mean=("sentiment","mean"), n_comments=("sentiment","size"))
           .reset_index())
    base = matches.rename(columns={"id":"match_id"})
    home = (agg
            .merge(base[["match_id","home"]], left_on=["match_id","team"], right_on=["match_id","home"], how="inner")
            .rename(columns={"sent_mean":"sent_home","n_comments":"n_home"})
            .drop(columns=["team","home"]))
    away = (agg
            .merge(base[["match_id","away"]], left_on=["match_id","team"], right_on=["match_id","away"], how="inner")
            .rename(columns={"sent_mean":"sent_away","n_comments":"n_away"})
            .drop(columns=["team","away"]))
    df = (base
          .merge(home, on="match_id", how="left")
          .merge(away, on="match_id", how="left"))
    df["n_home"] = df["n_home"].astype(float)
    df["n_away"] = df["n_away"].astype(float)
    df = df[(df["n_home"].fillna(0) >= min_comments) & (df["n_away"].fillna(0) >= min_comments)]
    if df.empty:
        return pd.DataFrame()
    df["sent_diff"] = df["sent_home"] - df["sent_away"]
    df["sent_mean"] = df[["sent_home","sent_away"]].mean(axis=1)
    df["target"] = df["result"].map({"H":0,"D":1,"A":2})
    df["total_comments"] = df["n_home"].fillna(0) + df["n_away"].fillna(0)
    return df.reset_index(drop=True)

def _select_probability_columns(df: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    mapping = {0:"p_h", 1:"p_d", 2:"p_a"}
    cols = [mapping[int(c)] for c in classes]
    probs = df[cols].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums

def _cv_probabilities(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> Optional[np.ndarray]:
    if np.unique(y).size < 2:
        return None
    counts = np.array([(y == c).sum() for c in classes])
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        return None
    min_class = positive_counts.min()
    if min_class < 2:
        return None
    n_splits = min(5, int(min_class))
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, multi_class="auto")
    try:
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    except Exception:
        return None

def _metric_summary(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray):
    idx_map = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([idx_map[val] for val in y_true])
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_idx] = 1
    return {
        "log_loss": float(log_loss(y_true, probs, labels=classes)),
        "brier": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "accuracy": float((classes[probs.argmax(axis=1)] == y_true).mean())
    }

@app.command("compare-predictors")
def compare_predictors(
    db: str = typer.Option("pl.db"),
    min_comments: int = typer.Option(15, help="Minimum pre-match comments per team required for evaluation")
):
    with connect(db) as conn:
        df = _build_sentiment_feature_frame(conn, max(1, int(min_comments)))
    if df.empty:
        typer.echo("No linked matches with enough sentiment data. Run scrape/analyze/link first or lower --min-comments.")
        raise typer.Exit()
    eval_df = df.dropna(subset=["target","sent_home","sent_away","sent_diff","p_h","p_d","p_a"])
    if eval_df.empty:
        typer.echo("No matches have both bookmaker odds and sentiment for the specified minimum comments.")
        raise typer.Exit()
    y = eval_df["target"].astype(int).to_numpy()
    classes = np.sort(np.unique(y))
    class_labels = ", ".join(["H" if c==0 else "D" if c==1 else "A" for c in classes])
    typer.echo(f"Matches evaluated: {len(eval_df)} | Outcomes present: {class_labels}")
    results = []
    odds_probs = _select_probability_columns(eval_df, classes)
    results.append(("Bookmaker implied probabilities", _metric_summary(y, odds_probs, classes)))
    sent_features = eval_df[["sent_home","sent_away","sent_diff","sent_mean"]].to_numpy(dtype=float)
    sent_probs = _cv_probabilities(sent_features, y, classes)
    if sent_probs is not None:
        results.append(("Sentiment (logistic CV)", _metric_summary(y, sent_probs, classes)))
    else:
        typer.echo("Not enough class balance to cross-validate sentiment-only model; skipping.")
    combo_features = eval_df[["sent_home","sent_away","sent_diff","sent_mean","p_h","p_d","p_a"]].to_numpy(dtype=float)
    combo_probs = _cv_probabilities(combo_features, y, classes)
    if combo_probs is not None:
        results.append(("Sentiment + odds (logistic CV)", _metric_summary(y, combo_probs, classes)))
    else:
        typer.echo("Not enough class balance to cross-validate combined model; skipping.")
    if results:
        header = f"{'Model':32s} {'LogLoss':>8s} {'Brier':>8s} {'Accuracy':>9s}"
        typer.echo(header)
        typer.echo("-" * len(header))
        for name, metrics in results:
            typer.echo(f"{name:32s} {metrics['log_loss']:8.3f} {metrics['brier']:8.3f} {metrics['accuracy']:9.3f}")
    preview_cols = ["date","home","away","result","p_h","p_d","p_a","sent_home","sent_away","sent_diff","n_home","n_away"]
    typer.echo("\nSample of matches used in comparison:")
    typer.echo(eval_df[preview_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    app()
