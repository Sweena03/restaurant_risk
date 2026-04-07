"""
agents/verification_agent.py
-----------------------------
Agent 6: Quality checks + human-in-the-loop flagging.

Checks:
  1. Sentiment coverage  — how many reviews had no keyword matches (all scores = 0)
  2. Label balance       — disengagement rate should be 15-60%
  3. Feature completeness — any critical features missing or all-NaN
  4. Strategy coverage   — any restaurants falling through to default strategy
"""

import pandas as pd
import numpy as np


class Verifier:

    def run(self, review_df: pd.DataFrame,
            restaurant_df: pd.DataFrame,
            action_df: pd.DataFrame) -> dict:

        issues, warnings, flagged = [], [], []

        self._check_sentiment_coverage(review_df, warnings)
        self._check_label_balance(restaurant_df, issues, warnings)
        self._check_feature_completeness(restaurant_df, warnings)
        self._check_strategy_coverage(action_df, flagged, warnings)

        passed  = len(issues) == 0
        summary = (
            f"{'✅ PASSED' if passed else '❌ FAILED'} — "
            f"{len(restaurant_df)} restaurants analysed, "
            f"{len(action_df)} flagged as at-risk. "
            f"{len(issues)} critical issue(s), {len(warnings)} warning(s)."
        )

        return {
            'passed':       passed,
            'issues':       issues,
            'warnings':     warnings,
            'flagged_rows': pd.DataFrame(flagged) if flagged else pd.DataFrame(),
            'summary':      summary,
        }

    def _check_sentiment_coverage(self, df, warnings):
        if not all(c in df.columns for c in
                   ['food_sentiment', 'service_sentiment', 'ambiance_sentiment']):
            return
        asp = ['food_sentiment', 'service_sentiment', 'ambiance_sentiment']
        zero_mask = df[asp].abs().max(axis=1) < 0.05
        pct = zero_mask.mean() * 100
        if pct > 40:
            warnings.append({
                'check':  'Sentiment Coverage',
                'detail': f'{pct:.1f}% of reviews had no aspect keyword matches. '
                          f'Scores defaulted to 0. Consider expanding keyword lists.'
            })

    def _check_label_balance(self, df, issues, warnings):
        if 'disengagement' not in df.columns:
            return
        rate = df['disengagement'].mean()
        if rate < 0.10:
            issues.append({
                'check':  'Label Balance',
                'detail': f'Only {rate*100:.1f}% disengagement — too few positives. '
                          f'Model will predict everything as healthy.'
            })
        elif rate > 0.70:
            issues.append({
                'check':  'Label Balance',
                'detail': f'{rate*100:.1f}% disengagement — too many positives. '
                          f'Check rating threshold in build_disengagement_label().'
            })
        else:
            warnings.append({
                'check':  'Label Balance',
                'detail': f'✅ Disengagement rate: {rate*100:.1f}% — within acceptable range.'
            })

    def _check_feature_completeness(self, df, warnings):
        for col in ['overall_rating', 'composite_sentiment', 'sentiment_trend']:
            if col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 20:
                    warnings.append({
                        'check':  'Feature Completeness',
                        'detail': f'{col} is missing for {missing_pct:.1f}% of restaurants.'
                    })

    def _check_strategy_coverage(self, action_df, flagged, warnings):
        if action_df.empty or 'strategy' not in action_df.columns:
            return
        default = 'Conduct a comprehensive review'
        default_mask = action_df['strategy'].str.startswith(default)
        n = default_mask.sum()
        if n > 0:
            warnings.append({
                'check':  'Strategy Coverage',
                'detail': f'{n} restaurants received the generic default strategy.'
            })
            rows = action_df[default_mask][
                ['restaurant_name', 'segment', 'dominant_complaint', 'strategy']
            ].copy()
            rows['flag_reason'] = 'Generic default strategy — review manually'
            flagged.extend(rows.to_dict('records'))
