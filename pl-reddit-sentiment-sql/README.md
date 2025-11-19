# PL Reddit Sentiment + Odds (SQLite)

Pipeline:
1) `init-db`
2) `scrape`
3) `load-odds`
4) `link-threads`
5) `analyze`
6) `app`
7) `compare-predictors` (optional)

Quickstart:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py init-db --db pl.db
python cli.py scrape "Arsenal, Tottenham, Liverpool" --db pl.db
python cli.py load-odds 2324 --db pl.db
python cli.py link-threads --db pl.db
python cli.py analyze --db pl.db
streamlit run app.py -- --db pl.db
# quantify odds vs sentiment predictive power
python cli.py compare-predictors --db pl.db
```
