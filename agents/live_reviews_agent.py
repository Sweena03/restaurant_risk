"""
agents/live_reviews_agent.py
-----------------------------
Fetches live restaurant review snippets via SerpApi Google Search.

Uses engine=google (free tier) to get snippets from Zomato, Swiggy,
TripAdvisor and Google's own review summaries.
Runs two searches per restaurant to get more content.
"""
import os
import pandas as pd

SERPAPI_KEY = "a20c36748813df03754ef4923fc29642237951f530afe6f064dc5ebcc92739de"

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False


def fetch_live_reviews(restaurant_name: str,
                       location: str = "Bangalore",
                       num_reviews: int = 20) -> pd.DataFrame:
    if not SERPAPI_AVAILABLE:
        print("[LiveReviews] serpapi not installed. Run: pip install google-search-results")
        return pd.DataFrame()

    key = SERPAPI_KEY or os.getenv("SERPAPI_KEY", "")
    if not key or key == "PASTE_YOUR_SERPAPI_KEY_HERE":
        print("[LiveReviews] No SerpApi key set.")
        return pd.DataFrame()

    # Two searches to get more content
    queries = [
        f"{restaurant_name} {location} restaurant reviews",
        f"{restaurant_name} {location} food service ambiance",
    ]

    records   = []
    name      = restaurant_name
    kg_rating = None

    for q in queries:
        print(f"[LiveReviews] Searching: {q}")
        try:
            results = GoogleSearch({
                "engine": "google",
                "q":      q,
                "api_key": key,
                "hl":     "en",
                "gl":     "in",
                "num":    10,
            }).get_dict()

            if results.get("error"):
                print(f"[LiveReviews] API error: {results['error']}")
                continue

            # Knowledge graph (first search only)
            kg = results.get("knowledge_graph", {})
            if kg and name == restaurant_name:
                name      = kg.get("title", restaurant_name)
                kg_rating = kg.get("rating")
                kg_desc   = kg.get("description", "")
                if kg_desc:
                    records.append({
                        "restaurant_name": name,
                        "review_text":     kg_desc,
                        "review_rating":   kg_rating,
                        "reviewer_name":   "Google Summary",
                        "source":          "google_kg",
                    })
                print(f"[LiveReviews] Found: {name} | Rating: {kg_rating}")

            # Organic snippets from Zomato, Swiggy, TripAdvisor etc.
            existing_texts = {r["review_text"] for r in records}
            for r in results.get("organic_results", []):
                if not isinstance(r, dict):
                    continue
                snippet = r.get("snippet", "").strip()
                if not snippet or len(snippet) < 25 or snippet in existing_texts:
                    continue
                existing_texts.add(snippet)
                records.append({
                    "restaurant_name": name,
                    "review_text":     snippet,
                    "review_rating":   r.get("rating") or kg_rating,
                    "reviewer_name":   r.get("source", "Web"),
                    "source":          "google_organic",
                })

            # People Also Ask
            for qa in results.get("related_questions", [])[:3]:
                if not isinstance(qa, dict):
                    continue
                ans = qa.get("snippet", "").strip()
                existing_texts = {r["review_text"] for r in records}
                if ans and len(ans) > 30 and ans not in existing_texts:
                    records.append({
                        "restaurant_name": name,
                        "review_text":     ans,
                        "review_rating":   kg_rating,
                        "reviewer_name":   "Google PAA",
                        "source":          "google_paa",
                    })

        except Exception as e:
            print(f"[LiveReviews] Search error: {e}")
            continue

    if not records:
        print(f"[LiveReviews] No content found for '{restaurant_name}'")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"[LiveReviews] ✅ {len(df)} snippets fetched for '{name}'")
    return df


def score_live_reviews(reviews_df: pd.DataFrame,
                       analyser,
                       predictor,
                       sentiment_df=None) -> dict:
    if reviews_df.empty:
        return {"error": "No data"}

    scored       = analyser.analyse_reviews(reviews_df)
    avg_food     = float(scored["food_sentiment"].mean())
    avg_service  = float(scored["service_sentiment"].mean())
    avg_ambiance = float(scored["ambiance_sentiment"].mean())
    avg_overall  = float(scored["overall_sentiment"].mean())
    avg_rating   = reviews_df["review_rating"].dropna().mean()

    from data.loader import CHURN_KEYWORDS, RETENTION_KEYWORDS
    texts       = reviews_df["review_text"].fillna("").str.lower()
    churn_count = int(texts.apply(lambda t: any(kw in t for kw in CHURN_KEYWORDS)).sum())
    ret_count   = int(texts.apply(lambda t: any(kw in t for kw in RETENTION_KEYWORDS)).sum())
    total       = len(reviews_df)

    if not pd.isna(avg_rating) and avg_rating < 3.0:
        risk, color = "High Risk",     "🔴"
    elif not pd.isna(avg_rating) and avg_rating < 3.8:
        risk, color = "Moderate Risk", "🟡"
    else:
        risk, color = "Low Risk",      "🟢"

    # Determine dominant complaint from sentiment scores
    asp_map = {
        "food_sentiment":     "Food",
        "service_sentiment":  "Service",
        "ambiance_sentiment": "Ambiance",
    }
    asp_scores = {
        "Food":     avg_food,
        "Service":  avg_service,
        "Ambiance": avg_ambiance,
    }
    dominant = min(asp_scores, key=asp_scores.get)
    if asp_scores[dominant] >= -0.2:
        dominant = "None"

    return {
        "restaurant_name":        reviews_df["restaurant_name"].iloc[0],
        "total_reviews":          total,
        "avg_rating":             round(float(avg_rating), 2) if not pd.isna(avg_rating) else None,
        "food_sentiment":         round(avg_food, 3),
        "service_sentiment":      round(avg_service, 3),
        "ambiance_sentiment":     round(avg_ambiance, 3),
        "overall_sentiment":      round(avg_overall, 3),
        "churn_intent_count":     churn_count,
        "retention_intent_count": ret_count,
        "risk_level":             risk,
        "risk_color":             color,
        "dominant_complaint":     dominant,
        "reviews":                scored,
    }
