import argparse, sqlite3, pandas as pd, numpy as np, streamlit as st
import datetime as dt

st.set_page_config(page_title="PL Reddit Sentiment + Odds (SQLite)", layout="wide")
st.title("Premier League: Reddit Sentiment + Odds")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--db", default="pl.db")
args, _ = parser.parse_known_args()
conn = sqlite3.connect(args.db)

def table_exists(name):
    return pd.read_sql_query("SELECT COUNT(1) n FROM sqlite_master WHERE type='table' AND name=?;", conn, params=(name,)).loc[0,'n']>0

threads_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM threads", conn).loc[0,'n'] if table_exists('threads') else 0
comments_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM comments", conn).loc[0,'n'] if table_exists('comments') else 0
st.metric("Threads", int(threads_n))
st.metric("Comments", int(comments_n))

if comments_n>0:
    st.subheader("Average sentiment by keyword")
    avg_kw = pd.read_sql_query("""
        SELECT keyword, AVG(sentiment) AS avg_sent, COUNT(*) AS n
        FROM comments WHERE sentiment IS NOT NULL
        GROUP BY keyword ORDER BY avg_sent DESC
    """, conn)
    st.dataframe(avg_kw)
    if not avg_kw.empty:
        st.bar_chart(avg_kw.set_index("keyword")["avg_sent"])

if comments_n>0:
    st.subheader("Top positive threads (>=10 comments)")
    top_threads = pd.read_sql_query("""
        SELECT c.thread_url, t.title, c.keyword,
               AVG(c.sentiment) AS avg_sent, COUNT(*) AS n_comments
        FROM comments c JOIN threads t ON t.url = c.thread_url
        WHERE c.sentiment IS NOT NULL
        GROUP BY c.thread_url, t.title, c.keyword
        HAVING COUNT(*) >= 10
        ORDER BY avg_sent DESC, n_comments DESC
        LIMIT 20
    """, conn)
    st.dataframe(top_threads)

if table_exists('matches'):
    st.subheader("Latest matches with implied probabilities")
    m = pd.read_sql_query("SELECT season, date, home, away, result, p_h, p_d, p_a FROM matches ORDER BY date DESC LIMIT 200", conn)
    st.dataframe(m)

if table_exists('matches') and threads_n>0 and comments_n>0:
    st.subheader("Linked threads → matches (pre-match sentiment vs odds)")
    df = pd.read_sql_query("""
        SELECT t.id as thread_id, t.title, t.match_id, t.created_utc,
               c.sentiment, c.created_utc as c_utc
        FROM threads t JOIN comments c ON c.thread_url = t.url
        WHERE t.match_id IS NOT NULL AND c.sentiment IS NOT NULL
    """, conn)
    if not df.empty:
        matches = pd.read_sql_query("SELECT id, date, home, away, p_h, p_d, p_a, result FROM matches", conn)
        matches["date_dt"] = pd.to_datetime(matches["date"])
        mapp = matches.set_index("id")["date_dt"].to_dict()
        df["match_date"] = df["match_id"].map(mapp)
        df["c_dt"] = pd.to_datetime(df["c_utc"], unit='s', errors='coerce')
        df_pre = df[df["c_dt"] <= (df["match_date"] + pd.Timedelta(hours=12))]
        agg = df_pre.groupby("thread_id").agg(avg_sent=("sentiment","mean")).reset_index()
        t2m = pd.read_sql_query("SELECT id as thread_id, match_id, title FROM threads WHERE match_id IS NOT NULL", conn)
        agg = agg.merge(t2m, on="thread_id", how="left")
        agg = agg.merge(matches.drop(columns=["date_dt"]), left_on="match_id", right_on="id", how="left")
        st.write("Pre-match avg sentiment per thread:")
        st.dataframe(agg[["title","date","home","away","result","avg_sent","p_h","p_d","p_a"]].sort_values("date", ascending=False).head(50))
        if not agg.empty:
            st.write("Scatter: pre-match sentiment vs implied Home win prob (p_h)")
            st.scatter_chart(agg[["avg_sent","p_h"]])
else:
    st.info("Load odds (load-odds) and link threads (link-threads) to enable this section.")
