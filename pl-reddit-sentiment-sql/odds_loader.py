import pandas as pd
import numpy as np
import requests
from io import BytesIO

def load_football_data_csv(season: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(BytesIO(r.content))
    df = df.rename(columns={
        "Date":"date", "HomeTeam":"home", "AwayTeam":"away", "FTR":"result",
        "B365H":"odds_h", "B365D":"odds_d", "B365A":"odds_a"
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date.astype(str)
    inv = 1/df[["odds_h","odds_d","odds_a"]].to_numpy(dtype=float)
    inv = inv / inv.sum(axis=1, keepdims=True)
    df[["p_h","p_d","p_a"]] = inv
    return df[["date","home","away","result","odds_h","odds_d","odds_a","p_h","p_d","p_a"]]
