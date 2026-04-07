"""
agents/risk_agent.py
---------------------
Agent 3: Restaurant Health Index (RHI) — replaces traditional RFM.

Since the Zomato dataset has no individual customer transaction logs,
traditional RFM (Recency/Frequency/Monetary per customer) is not applicable.

Instead, we compute a Restaurant Health Index with 5 dimensions:
  1. Rating Health    — overall rating vs area average
  2. Sentiment Health — NLP-derived food/service/ambiance scores  
  3. Engagement       — votes relative to price tier
  4. Value Perception — rating vs price (are customers getting value?)
  5. Trend            — is sentiment improving or declining over reviews?

Each dimension scored 1-4. Sum = RHI score (5-20).
Higher = healthier restaurant with lower disengagement risk.
"""

import pandas as pd
import numpy as np


SEGMENT_LABELS = {
    (17, 20): 'Thriving',
    (13, 16): 'Stable',
    (9,  12): 'At-Risk',
    (5,   8): 'Critical',
}


def _safe_qcut(series: pd.Series, q: int = 4) -> pd.Series:
    """Quantile cut with safety fallback for duplicate edges."""
    try:
        return pd.qcut(
            series.rank(method='first'),
            q=q, labels=False, duplicates='drop'
        ).fillna(0).astype(int) + 1
    except Exception:
        return pd.Series([2] * len(series), index=series.index)


def compute_health_index(
    rfm_df:       pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges RFM-style engagement data with NLP sentiment data
    and computes the Restaurant Health Index per restaurant.

    Args:
        rfm_df       : from risk_agent — has votes, price_for_two, overall_rating etc.
        sentiment_df : from nlp_agent  — has sentiment scores per restaurant

    Returns:
        DataFrame with RHI score, segment, and all component scores.
    """
    df = rfm_df.merge(sentiment_df, on='restaurant_name', how='left')

    # ── Dimension 1: Rating Health ──────────────────────────────────────────
    # How does this restaurant's rating compare to others in the same area?
    area_avg = df.groupby('location')['overall_rating'].transform('mean')
    df['rating_vs_area'] = df['overall_rating'].fillna(3.5) - area_avg.fillna(3.5)
    df['d1_rating'] = _safe_qcut(df['rating_vs_area'])

    # ── Dimension 2: Sentiment Health ───────────────────────────────────────
    # Average of all three NLP aspect scores
    df['composite_sentiment'] = df[
        ['food_sentiment', 'service_sentiment', 'ambiance_sentiment']
    ].mean(axis=1).fillna(0)
    df['d2_sentiment'] = _safe_qcut(df['composite_sentiment'])

    # ── Dimension 3: Engagement ─────────────────────────────────────────────
    # Votes normalised within price tier (cheap restaurant with 1000 votes
    # is more engaged than expensive restaurant with same votes)
    df['price_tier'] = pd.qcut(
        df['price_for_two'].fillna(df['price_for_two'].median()),
        q=4, labels=['Budget', 'Mid', 'Premium', 'Luxury'], duplicates='drop'
    )
    tier_avg_votes  = df.groupby('price_tier')['votes'].transform('mean')
    df['votes_norm'] = df['votes'] / tier_avg_votes.replace(0, 1)
    df['d3_engagement'] = _safe_qcut(df['votes_norm'].fillna(1))

    # ── Dimension 4: Value Perception ───────────────────────────────────────
    # Customers feel value if rating is high relative to price
    # Normalise: rating / log(price+1)
    df['value_score'] = df['overall_rating'].fillna(3.5) / np.log1p(
        df['price_for_two'].fillna(500)
    )
    df['d4_value'] = _safe_qcut(df['value_score'])

    # ── Dimension 5: Trend ──────────────────────────────────────────────────
    # Is sentiment improving or declining over time?
    df['d5_trend'] = _safe_qcut(df['sentiment_trend'].fillna(0))

    # ── RHI Score ────────────────────────────────────────────────────────────
    for col in ['d1_rating', 'd2_sentiment', 'd3_engagement', 'd4_value', 'd5_trend']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(2).astype(int)

    df['rhi_score'] = (
        df['d1_rating'] + df['d2_sentiment'] + df['d3_engagement'] +
        df['d4_value']  + df['d5_trend']
    )

    # ── Segment ───────────────────────────────────────────────────────────────
    def assign_segment(score):
        if score >= 17: return 'Thriving'
        elif score >= 13: return 'Stable'
        elif score >= 9:  return 'At-Risk'
        else:             return 'Critical'

    df['segment'] = df['rhi_score'].apply(assign_segment)

    print(f'[RiskAgent] Health Index computed for {len(df)} restaurants')
    print(df['segment'].value_counts().to_string())

    return df


def compute_restaurant_features(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates review-level data to restaurant level.
    This feeds into compute_health_index().
    """
    agg = reviews_df.groupby('restaurant_name').agg(
        overall_rating = ('overall_rating', 'first'),
        price_for_two  = ('price_for_two',  'first'),
        votes          = ('votes',           'first'),
        location       = ('location',        'first'),
        cuisines       = ('cuisines',        'first'),
        online_order   = ('online_order',    'first'),
        book_table     = ('book_table',      'first'),
        review_count   = ('review_text',     'count'),
        churn_rate     = ('churn',           'mean'),
        avg_review_rating = ('review_rating','mean'),
        explicit_churn = ('label_source',
                          lambda x: (x == 'explicit_churn').sum()),
        explicit_retention = ('label_source',
                               lambda x: (x == 'explicit_retention').sum()),
    ).reset_index()

    # Churn intent ratio from explicit labels only
    total_explicit = agg['explicit_churn'] + agg['explicit_retention']
    agg['explicit_churn_ratio'] = (
        agg['explicit_churn'] / total_explicit.replace(0, np.nan)
    ).fillna(0)

    print(f'[RiskAgent] Aggregated {len(agg)} restaurants from reviews')
    return agg
