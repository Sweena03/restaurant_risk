"""
agents/strategy_agent.py
-------------------------
Agent 5: Generates actionable improvement strategies per restaurant
based on their segment, dominant complaint, and SHAP explanation.

Strategy matrix: 4 segments × 4 complaint types = 16 combinations
plus trend-aware modifiers.
"""

import pandas as pd

STRATEGIES = {
    # ── CRITICAL ─────────────────────────────────────────────────────────────
    ('Critical', 'Food'): (
        'Immediate menu overhaul needed. Consider bringing in a consultant chef. '
        'Run a "New Menu" campaign with free tasting events to rebuild trust. '
        'Address specific dishes mentioned in negative reviews.'
    ),
    ('Critical', 'Service'): (
        'Emergency staff retraining required. Hire a floor manager if none exists. '
        'Implement service standards and conduct weekly reviews. '
        'Respond publicly to negative reviews with an action plan.'
    ),
    ('Critical', 'Ambiance'): (
        'Invest in immediate visible improvements — deep cleaning, lighting, seating. '
        'Consider a temporary closure for renovation if feasible. '
        'Announce improvements on social media to signal change.'
    ),
    ('Critical', 'None'): (
        'Multiple issues detected with no single dominant complaint. '
        'Conduct an urgent customer feedback survey. '
        'Review all operations: menu, staffing, pricing, and cleanliness.'
    ),

    # ── AT-RISK ───────────────────────────────────────────────────────────────
    ('At-Risk', 'Food'): (
        'Refresh 20-30% of menu items based on most-complained dishes. '
        'Highlight quality ingredients in marketing. '
        'Offer a "Chef Recommends" section with newer, better-reviewed items.'
    ),
    ('At-Risk', 'Service'): (
        'Introduce table check-ins (staff asks "How is everything?" mid-meal). '
        'Reduce wait times with better kitchen coordination. '
        'Train staff on handling complaints gracefully.'
    ),
    ('At-Risk', 'Ambiance'): (
        'Low-cost ambiance upgrades: plants, lighting, background music playlist. '
        'Address noise complaints with soft furnishings or table spacing. '
        'Promote quieter time slots for customers who prefer calm dining.'
    ),
    ('At-Risk', 'None'): (
        'Declining engagement without a clear complaint — likely convenience or value issue. '
        'Review pricing against competitors in the area. '
        'Launch a loyalty offer to re-engage lapsed customers.'
    ),

    # ── STABLE ────────────────────────────────────────────────────────────────
    ('Stable', 'Food'): (
        'Good overall health but food scores lagging. '
        'Introduce seasonal specials and limited-time dishes to refresh interest. '
        'Promote best-reviewed dishes more prominently.'
    ),
    ('Stable', 'Service'): (
        'Stable restaurant with service friction. '
        'Peak-hour staffing may be the issue — add staff during busy periods. '
        'Introduce reservation system to manage flow better.'
    ),
    ('Stable', 'Ambiance'): (
        'Stable restaurant with ambiance complaints. '
        'Small targeted upgrades in the complained area (noise/lighting/cleanliness). '
        'Promote outdoor seating or quieter sections if available.'
    ),
    ('Stable', 'None'): (
        'Restaurant is stable with no dominant complaint. '
        'Focus on building loyalty: introduce a stamp card or repeat-visit reward. '
        'Encourage satisfied customers to leave reviews to boost visibility.'
    ),

    # ── THRIVING ──────────────────────────────────────────────────────────────
    ('Thriving', 'Food'): (
        'Thriving restaurant — protect food quality as the core asset. '
        'Consider expanding the menu or adding a signature dish. '
        'Use food quality as the main marketing message.'
    ),
    ('Thriving', 'Service'): (
        'Thriving restaurant — minor service inconsistency. '
        'Maintain current standards and address specific staff feedback. '
        'Use your strong reputation to attract and retain quality staff.'
    ),
    ('Thriving', 'Ambiance'): (
        'Thriving restaurant with ambiance as the only weak point. '
        'Invest profits into a targeted ambiance upgrade. '
        'This could push the restaurant into top-rated territory.'
    ),
    ('Thriving', 'None'): (
        'Top-performing restaurant. '
        'Focus on maintaining consistency and expanding reach. '
        'Consider opening a second location or catering service.'
    ),
}

DEFAULT_STRATEGY = (
    'Conduct a comprehensive review of customer feedback. '
    'Focus on the aspect with the lowest sentiment score. '
    'Set measurable improvement targets for the next 90 days.'
)


def generate_strategy(segment: str, complaint: str) -> str:
    return STRATEGIES.get((segment, complaint), DEFAULT_STRATEGY)


def apply_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Adds strategy column to restaurant-level DataFrame."""
    df = df.copy()
    df['strategy'] = df.apply(
        lambda row: generate_strategy(
            str(row.get('segment', '')),
            str(row.get('dominant_complaint', 'None'))
        ),
        axis=1,
    )

    # Add trend modifier note
    def trend_note(trend):
        if pd.isna(trend):
            return ''
        if trend < -0.2:
            return ' ⚠️ Sentiment declining — act urgently.'
        elif trend > 0.2:
            return ' ✅ Sentiment improving — keep momentum.'
        return ''

    if 'sentiment_trend' in df.columns:
        df['strategy'] = df['strategy'] + df['sentiment_trend'].apply(trend_note)

    return df


def build_action_list(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Returns top_n highest-risk restaurants sorted by risk_probability."""
    if 'risk_probability' not in df.columns:
        return df.head(top_n)

    at_risk = df[df['risk_probability'] >= 0.40].copy()
    at_risk = at_risk.sort_values('risk_probability', ascending=False)

    cols = [
        'restaurant_name', 'location', 'segment', 'risk_probability',
        'risk_level', 'overall_rating', 'dominant_complaint',
        'shap_explanation', 'strategy'
    ]
    available = [c for c in cols if c in at_risk.columns]
    return at_risk[available].head(top_n).reset_index(drop=True)
