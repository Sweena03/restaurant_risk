"""
data/loader.py
--------------
Loads and merges:
  1. zomato_bangalore_historical.csv — main training data (Kaggle: himanshupoddar)
  2. zomato_recent_2024_2025.csv     — recent data, same schema (Kaggle: teshavsharma)

Swiggy excluded — different column schema, no review text, would break NLP.

Churn labelling strategy:
  Explicit : review contains known churn/retention keywords → reliable label
  Soft     : review rating <= 3.0 → inferred label
  Validated: churn reviews avg 1.79 stars vs 4.52 for retention (EDA confirmed)
"""

import re, ast, hashlib, os
import pandas as pd
import numpy as np

CHURN_KEYWORDS = [
    "not visiting again", "will not return", "never coming back",
    "won't be back", "definitely not visiting", "not coming back",
    "never again", "avoid this", "stay away", "not recommended",
    "don't go", "wont visit", "never visit", "worst experience",
    "definitely not recommend", "never recommend", "not worth",
    "waste of", "pathetic", "terrible", "horrible"
]

RETENTION_KEYWORDS = [
    "will visit again", "would visit again", "coming back",
    "definitely returning", "will return", "would return",
    "highly recommend", "must visit", "visit again soon",
    "surely like to come", "will come again", "would come again",
    "definitely going back", "surely visit", "surely come",
    "would definitely", "highly recommended", "loved it",
    "amazing place", "best restaurant"
]


def detect_intent(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in CHURN_KEYWORDS):
        return 'churn'
    elif any(kw in t for kw in RETENTION_KEYWORDS):
        return 'retention'
    return 'neutral'


def parse_rate(v) -> float:
    try:
        return float(str(v).replace('/5', '').strip())
    except (ValueError, TypeError):
        return np.nan


def parse_cost(v) -> float:
    try:
        return float(str(v).replace(',', '').strip())
    except (ValueError, TypeError):
        return np.nan


def _parse_zomato_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        overall_rating = parse_rate(row.get('rate', np.nan))
        price_for_two  = parse_cost(row.get('approx_cost(for two people)', np.nan))
        name           = str(row.get('name', 'Unknown')).strip()
        location       = str(row.get('location', row.get('listed_in(city)', 'Unknown')))
        votes          = int(row.get('votes', 0)) if pd.notna(row.get('votes', 0)) else 0
        restaurant_id  = hashlib.sha256(name.encode()).hexdigest()[:12]

        try:
            reviews = ast.literal_eval(str(row.get('reviews_list', '[]')))
        except Exception:
            reviews = []

        if not reviews:
            records.append({
                'restaurant_id': restaurant_id, 'restaurant_name': name,
                'location': location, 'cuisines': str(row.get('cuisines', '')),
                'overall_rating': overall_rating, 'price_for_two': price_for_two,
                'votes': votes, 'online_order': str(row.get('online_order', 'No')),
                'book_table': str(row.get('book_table', 'No')),
                'review_text': '', 'review_rating': overall_rating,
                'intent': 'neutral',
                'churn': 1 if (overall_rating or 5) < 3.5 else 0,
                'label_source': 'overall_rating', 'review_index': 0, '_source': source,
            })
            continue

        for idx, item in enumerate(reviews):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            rating_str    = str(item[0])
            review_text   = str(item[1]).replace('RATED\n', '').strip()
            match         = re.search(r'(\d+\.?\d*)', rating_str)
            review_rating = float(match.group(1)) if match else overall_rating
            intent        = detect_intent(review_text)

            if intent == 'churn':
                churn, ls = 1, 'explicit_churn'
            elif intent == 'retention':
                churn, ls = 0, 'explicit_retention'
            else:
                churn, ls = (1 if (review_rating or 5) <= 3.0 else 0), 'soft_rating'

            records.append({
                'restaurant_id': restaurant_id, 'restaurant_name': name,
                'location': location, 'cuisines': str(row.get('cuisines', '')),
                'overall_rating': overall_rating, 'price_for_two': price_for_two,
                'votes': votes, 'online_order': str(row.get('online_order', 'No')),
                'book_table': str(row.get('book_table', 'No')),
                'review_text': review_text,
                'review_rating': review_rating if review_rating else overall_rating,
                'intent': intent, 'churn': churn, 'label_source': ls,
                'review_index': idx, '_source': source,
            })
    return pd.DataFrame(records)


def load_zomato(path: str, nrows: int = None, recent_path: str = None) -> pd.DataFrame:
    """
    Load and parse Zomato dataset(s) into flat review-level DataFrame.

    Args:
        path        : Path to historical Zomato CSV
        nrows       : Row limit for historical data (None = all)
        recent_path : Optional path to recent Zomato CSV (same schema)
    """
    print('[Loader] Reading historical Zomato...')
    df_hist = pd.read_csv(path, nrows=nrows)
    print(f'[Loader] {len(df_hist):,} historical restaurants')
    frames = [_parse_zomato_df(df_hist, 'zomato_historical')]

    if recent_path and os.path.exists(recent_path):
        print('[Loader] Reading recent Zomato...')
        df_recent = pd.read_csv(recent_path)
        frames.append(_parse_zomato_df(df_recent, 'zomato_recent'))
        print(f'[Loader] {len(df_recent):,} recent restaurants added')
    else:
        if recent_path:
            print(f'[Loader] Recent Zomato not found at {recent_path} — using historical only')

    result   = pd.concat(frames, ignore_index=True)
    explicit = result['label_source'].str.startswith('explicit').sum()
    soft     = result['label_source'].str.startswith('soft').sum()
    print(f'[Loader] Total: {len(result):,} reviews | '
          f'{result["restaurant_name"].nunique():,} restaurants')
    print(f'  Explicit: {explicit:,} | Soft: {soft:,} | '
          f'Churn rate: {result["churn"].mean()*100:.1f}%')
    return result
