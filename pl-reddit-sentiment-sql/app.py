import argparse, sqlite3, pandas as pd, numpy as np, streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="PL Reddit Sentiment + Odds", layout="wide")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--db", default="pl.db")
args, _ = parser.parse_known_args()

# For deployment: use bundled pl.db in same directory as app.py
DB_PATH = Path(__file__).parent / args.db
if not DB_PATH.exists():
    DB_PATH = Path(args.db)  # fallback to relative path

@st.cache_resource
def get_connection():
    """Cached DB connection for performance."""
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

conn = get_connection()

# --- Helper functions ---
@st.cache_data(ttl=600)
def table_exists(name):
    return pd.read_sql_query("SELECT COUNT(1) n FROM sqlite_master WHERE type='table' AND name=?;", get_connection(), params=(name,)).loc[0,'n'] > 0

@st.cache_data(ttl=600)
def get_counts():
    """Get thread/comment/match counts (cached)."""
    conn = get_connection()
    threads_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM threads", conn).loc[0,'n'] if table_exists('threads') else 0
    comments_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM comments", conn).loc[0,'n'] if table_exists('comments') else 0
    matches_n = pd.read_sql_query("SELECT COUNT(*) AS n FROM matches", conn).loc[0,'n'] if table_exists('matches') else 0
    return int(threads_n), int(comments_n), int(matches_n)

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

def canonicalize(token):
    if not isinstance(token, str):
        return None
    token = ALIASES.get(token, token)
    for t in TEAMS:
        if t.lower() in token.lower():
            return t
    return None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Team Analysis", "Matches & Odds", "Raw Data"])

# --- Data loading ---
threads_n, comments_n, matches_n = get_counts()

# ============================================================
# PAGE: Overview
# ============================================================
if page == "Overview":
    st.title("🏆 Premier League: Reddit Sentiment + Odds")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Threads", int(threads_n))
    col2.metric("💬 Comments", int(comments_n))
    col3.metric("⚽ Matches", int(matches_n))
    
    if comments_n > 0:
        st.subheader("Average Sentiment by Keyword")
        avg_kw = pd.read_sql_query("""
            SELECT keyword, AVG(sentiment) AS avg_sent, COUNT(*) AS n
            FROM comments WHERE sentiment IS NOT NULL
            GROUP BY keyword ORDER BY avg_sent DESC
        """, conn)
        if not avg_kw.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in avg_kw['avg_sent']]
            ax.barh(avg_kw['keyword'], avg_kw['avg_sent'], color=colors)
            ax.set_xlabel('Average Sentiment')
            ax.set_title('Sentiment by Search Keyword')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(avg_kw, use_container_width=True)
    
    if comments_n > 0:
        st.subheader("Top Positive Threads (≥10 comments)")
        top_threads = pd.read_sql_query("""
            SELECT c.thread_url, t.title, c.keyword,
                   AVG(c.sentiment) AS avg_sent, COUNT(*) AS n_comments
            FROM comments c JOIN threads t ON t.url = c.thread_url
            WHERE c.sentiment IS NOT NULL
            GROUP BY c.thread_url, t.title, c.keyword
            HAVING COUNT(*) >= 10
            ORDER BY avg_sent DESC, n_comments DESC
            LIMIT 15
        """, conn)
        if not top_threads.empty:
            st.dataframe(top_threads, use_container_width=True)

# ============================================================
# PAGE: Team Analysis
# ============================================================
elif page == "Team Analysis":
    st.title("📊 Team Sentiment Profiles")
    st.markdown("Aggregate sentiment per team correlated with historical performance.")
    st.markdown("---")
    
    if comments_n == 0:
        st.warning("No comments found. Run `scrape` and `analyze` first.")
    else:
        # Compute team sentiment
        comments_df = pd.read_sql_query(
            "SELECT keyword, sentiment FROM comments WHERE sentiment IS NOT NULL", conn
        )
        comments_df["team"] = comments_df["keyword"].apply(canonicalize)
        comments_df = comments_df.dropna(subset=["team"])
        
        if comments_df.empty:
            st.warning("No team-mappable comments found.")
        else:
            team_sentiment = (comments_df
                .groupby("team")
                .agg(
                    avg_sentiment=("sentiment", "mean"),
                    median_sentiment=("sentiment", "median"),
                    std_sentiment=("sentiment", "std"),
                    n_comments=("sentiment", "size"),
                    positive_pct=("sentiment", lambda x: (x > 0.05).mean() * 100),
                    negative_pct=("sentiment", lambda x: (x < -0.05).mean() * 100),
                )
                .reset_index()
                .sort_values("avg_sentiment", ascending=False))
            
            st.subheader("Sentiment Rankings")
            
            # Sentiment bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['#27ae60' if x >= 0 else '#c0392b' for x in team_sentiment['avg_sentiment']]
            bars = ax.barh(team_sentiment['team'], team_sentiment['avg_sentiment'], color=colors)
            ax.set_xlabel('Average Sentiment Score', fontsize=12)
            ax.set_title('Team Sentiment Rankings (Reddit Comments)', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2)
            ax.set_xlim(-0.5, 0.5)
            for i, (sent, n) in enumerate(zip(team_sentiment['avg_sentiment'], team_sentiment['n_comments'])):
                ax.annotate(f'n={int(n)}', xy=(sent + 0.02 if sent >= 0 else sent - 0.08, i), 
                           va='center', fontsize=8, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Sentiment table
            st.dataframe(
                team_sentiment.style.format({
                    'avg_sentiment': '{:.3f}',
                    'median_sentiment': '{:.3f}',
                    'std_sentiment': '{:.3f}',
                    'positive_pct': '{:.1f}%',
                    'negative_pct': '{:.1f}%',
                }),
                use_container_width=True
            )
            
            # Sentiment distribution (box plot)
            st.subheader("Sentiment Distribution by Team")
            teams_with_data = comments_df["team"].value_counts()
            top_teams = teams_with_data[teams_with_data >= 10].index.tolist()[:10]
            
            if top_teams:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                box_data = [comments_df[comments_df["team"] == t]["sentiment"].values for t in top_teams]
                bp = ax2.boxplot(box_data, labels=top_teams, patch_artist=True, vert=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('#3498db')
                    patch.set_alpha(0.6)
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax2.set_ylabel('Sentiment Score', fontsize=12)
                ax2.set_xlabel('Team', fontsize=12)
                ax2.set_title('Sentiment Distribution (Top 10 Teams by Comment Volume)', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Historical form correlation
            if table_exists('matches') and matches_n > 0:
                st.subheader("Sentiment vs Historical Performance")
                
                matches = pd.read_sql_query("SELECT home, away, result FROM matches", conn)
                
                team_form = {}
                for team in team_sentiment['team'].unique():
                    home_matches = matches[matches["home"] == team]
                    away_matches = matches[matches["away"] == team]
                    
                    home_wins = (home_matches["result"] == "H").sum()
                    away_wins = (away_matches["result"] == "A").sum()
                    total = len(home_matches) + len(away_matches)
                    wins = home_wins + away_wins
                    draws = (home_matches["result"] == "D").sum() + (away_matches["result"] == "D").sum()
                    
                    if total > 0:
                        team_form[team] = {
                            "team": team,
                            "matches": total,
                            "wins": wins,
                            "draws": draws,
                            "losses": total - wins - draws,
                            "win_pct": wins / total * 100,
                            "ppg": (wins * 3 + draws) / total,
                        }
                
                form_df = pd.DataFrame(list(team_form.values()))
                
                if not form_df.empty:
                    merged = team_sentiment.merge(form_df, on="team", how="inner")
                    
                    if len(merged) >= 3:
                        # Scatter: sentiment vs win %
                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        scatter = ax3.scatter(
                            merged['avg_sentiment'], 
                            merged['win_pct'],
                            s=merged['n_comments'] / merged['n_comments'].max() * 500 + 50,
                            c=merged['ppg'],
                            cmap='RdYlGn',
                            alpha=0.7,
                            edgecolors='black'
                        )
                        
                        for i, row in merged.iterrows():
                            ax3.annotate(row['team'], (row['avg_sentiment'], row['win_pct']),
                                        fontsize=8, ha='center', va='bottom')
                        
                        ax3.set_xlabel('Average Reddit Sentiment', fontsize=12)
                        ax3.set_ylabel('Historical Win %', fontsize=12)
                        ax3.set_title('Sentiment vs Performance (size=comments, color=PPG)', fontsize=14, fontweight='bold')
                        plt.colorbar(scatter, label='Points Per Game')
                        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        
                        # Correlation stats
                        corr_sent_win = merged['avg_sentiment'].corr(merged['win_pct'])
                        corr_sent_ppg = merged['avg_sentiment'].corr(merged['ppg'])
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Correlation: Sentiment ↔ Win%", f"{corr_sent_win:.3f}")
                        col2.metric("Correlation: Sentiment ↔ PPG", f"{corr_sent_ppg:.3f}")
                        
                        st.caption("Positive correlation = fans of winning teams are more positive. Negative = opposite.")
                        
                        # Combined table
                        st.subheader("Combined Profile")
                        display_cols = ["team", "avg_sentiment", "n_comments", "positive_pct", "matches", "wins", "win_pct", "ppg"]
                        st.dataframe(
                            merged[display_cols].sort_values("ppg", ascending=False).style.format({
                                'avg_sentiment': '{:.3f}',
                                'positive_pct': '{:.1f}%',
                                'win_pct': '{:.1f}%',
                                'ppg': '{:.2f}',
                            }),
                            use_container_width=True
                        )
            
            # Sentiment by match outcome (if linked threads exist)
            linked_threads = pd.read_sql_query("""
                SELECT t.keyword, c.sentiment, m.result, m.home, m.away
                FROM threads t 
                JOIN comments c ON c.thread_url = t.url
                JOIN matches m ON m.id = t.match_id
                WHERE t.match_id IS NOT NULL AND c.sentiment IS NOT NULL
            """, conn)
            
            if not linked_threads.empty:
                st.subheader("Sentiment by Match Outcome")
                st.caption("How does fan sentiment vary when their team wins/draws/loses?")
                
                linked_threads["team"] = linked_threads["keyword"].apply(canonicalize)
                linked_threads = linked_threads.dropna(subset=["team"])
                
                def team_outcome(row):
                    if row["team"] == row["home"]:
                        return {"H": "Win", "D": "Draw", "A": "Loss"}.get(row["result"], "Unknown")
                    elif row["team"] == row["away"]:
                        return {"A": "Win", "D": "Draw", "H": "Loss"}.get(row["result"], "Unknown")
                    return "Unknown"
                
                linked_threads["outcome"] = linked_threads.apply(team_outcome, axis=1)
                linked_threads = linked_threads[linked_threads["outcome"] != "Unknown"]
                
                if not linked_threads.empty:
                    outcome_agg = linked_threads.groupby("outcome")["sentiment"].agg(["mean", "count"]).reset_index()
                    outcome_agg.columns = ["Outcome", "Avg Sentiment", "Comments"]
                    
                    fig4, ax4 = plt.subplots(figsize=(8, 5))
                    colors_outcome = {'Win': '#27ae60', 'Draw': '#f39c12', 'Loss': '#c0392b'}
                    bars = ax4.bar(outcome_agg['Outcome'], outcome_agg['Avg Sentiment'], 
                                  color=[colors_outcome.get(o, 'gray') for o in outcome_agg['Outcome']])
                    ax4.axhline(y=0, color='gray', linestyle='--')
                    ax4.set_ylabel('Average Sentiment')
                    ax4.set_title('Fan Sentiment by Match Outcome', fontsize=14, fontweight='bold')
                    for bar, n in zip(bars, outcome_agg['Comments']):
                        ax4.annotate(f'n={int(n)}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                    ha='center', va='bottom', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig4)

# ============================================================
# PAGE: Matches & Odds
# ============================================================
elif page == "Matches & Odds":
    st.title("⚽ Matches & Bookmaker Odds")
    st.markdown("---")
    
    if not table_exists('matches') or matches_n == 0:
        st.warning("No matches loaded. Run `load-odds <season>` first.")
    else:
        # Season filter
        seasons = pd.read_sql_query("SELECT DISTINCT season FROM matches ORDER BY season DESC", conn)['season'].tolist()
        selected_season = st.selectbox("Select Season", ["All"] + seasons)
        
        query = "SELECT season, date, home, away, result, p_h, p_d, p_a FROM matches"
        if selected_season != "All":
            query += f" WHERE season = '{selected_season}'"
        query += " ORDER BY date DESC LIMIT 200"
        
        m = pd.read_sql_query(query, conn)
        
        st.subheader("Latest Matches with Implied Probabilities")
        st.dataframe(
            m.style.format({'p_h': '{:.2%}', 'p_d': '{:.2%}', 'p_a': '{:.2%}'}),
            use_container_width=True
        )
        
        # Outcome distribution
        st.subheader("Outcome Distribution")
        outcome_counts = m['result'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {'H': '#27ae60', 'D': '#f39c12', 'A': '#c0392b'}
        ax.pie(outcome_counts.values, labels=[f"{k} ({v})" for k, v in outcome_counts.items()],
               colors=[colors.get(k, 'gray') for k in outcome_counts.index],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Match Outcomes')
        st.pyplot(fig)

# ============================================================
# PAGE: Raw Data
# ============================================================
elif page == "Raw Data":
    st.title("🗃️ Raw Data Explorer")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Threads", "Comments", "Matches"])
    
    with tab1:
        if table_exists('threads'):
            st.dataframe(pd.read_sql_query("SELECT * FROM threads ORDER BY created_utc DESC LIMIT 100", conn), use_container_width=True)
        else:
            st.info("No threads table.")
    
    with tab2:
        if table_exists('comments'):
            st.dataframe(pd.read_sql_query("SELECT id, thread_url, keyword, author, body, sentiment FROM comments WHERE sentiment IS NOT NULL ORDER BY id DESC LIMIT 100", conn), use_container_width=True)
        else:
            st.info("No comments table.")
    
    with tab3:
        if table_exists('matches'):
            st.dataframe(pd.read_sql_query("SELECT * FROM matches ORDER BY date DESC LIMIT 100", conn), use_container_width=True)
        else:
            st.info("No matches table.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"DB: {args.db}")
st.sidebar.caption(f"Threads: {threads_n} | Comments: {comments_n} | Matches: {matches_n}")
