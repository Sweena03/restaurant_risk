"""
agents/prediction_agent.py
---------------------------
Agent 4: Disengagement Prediction

Academic improvements:
  - overall_rating excluded from features (used in label = data leakage)
  - SMOTE oversampling for class imbalance
  - 5-fold stratified cross-validation
  - Baseline comparison: Logistic Regression + Random Forest vs XGBoost
  - Full metrics: accuracy, F1, precision, recall, ROC-AUC
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle, os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, f1_score, precision_score,
                                      recall_score, classification_report, roc_auc_score)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# overall_rating intentionally excluded — it directly determines the label
# (label = rating < 3.5) so including it causes 100% accuracy via leakage
FEATURE_COLS = [
    'd2_sentiment', 'd3_engagement', 'd4_value', 'd5_trend',
    'composite_sentiment', 'sentiment_trend',
    'food_sentiment', 'service_sentiment', 'ambiance_sentiment',
    'explicit_churn_ratio', 'votes', 'review_count',
]


def build_disengagement_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    churn_col  = (df['churn_rate'] > 0.35
                  if 'churn_rate' in df.columns
                  else pd.Series([False] * len(df), index=df.index))
    rating_col = df['overall_rating'].fillna(5) < 3.5
    df['disengagement'] = (churn_col | rating_col).astype(int)
    rate = df['disengagement'].mean()
    print(f'[PredictionAgent] Disengagement rate: {rate*100:.1f}% ({df["disengagement"].sum()} / {len(df)})')
    return df


def _run_baselines(X_tr, X_te, y_tr, y_te) -> dict:
    baselines = {}

    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))])
    lr.fit(X_tr, y_tr)
    p = lr.predict(X_te)
    baselines['Logistic Regression'] = {
        'accuracy':  round(accuracy_score(y_te, p), 4),
        'f1':        round(f1_score(y_te, p, pos_label=1), 4),
        'precision': round(precision_score(y_te, p, pos_label=1, zero_division=0), 4),
        'recall':    round(recall_score(y_te, p, pos_label=1), 4),
    }

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_tr, y_tr)
    p = rf.predict(X_te)
    baselines['Random Forest'] = {
        'accuracy':  round(accuracy_score(y_te, p), 4),
        'f1':        round(f1_score(y_te, p, pos_label=1), 4),
        'precision': round(precision_score(y_te, p, pos_label=1, zero_division=0), 4),
        'recall':    round(recall_score(y_te, p, pos_label=1), 4),
    }

    print('\n[PredictionAgent] Baseline comparison:')
    for name, m in baselines.items():
        print(f'  {name:22s} Acc:{m["accuracy"]*100:.1f}% F1:{m["f1"]*100:.1f}% '
              f'Prec:{m["precision"]*100:.1f}% Recall:{m["recall"]*100:.1f}%')
    return baselines


def _run_cross_validation(model, X, y) -> dict:
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=cv,
                              scoring=['accuracy','f1','precision','recall','roc_auc'],
                              return_train_score=False)
    cv_out = {}
    print('\n[PredictionAgent] 5-Fold Cross-Validation:')
    for metric in ['accuracy','f1','precision','recall','roc_auc']:
        scores = results[f'test_{metric}']
        cv_out[f'{metric}_mean'] = round(float(scores.mean()), 4)
        cv_out[f'{metric}_std']  = round(float(scores.std()),  4)
        print(f'  {metric:12s}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%')
    return cv_out


class DisengagementPredictor:

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2,
            use_label_encoder=False, eval_metric='logloss', random_state=42,
        )
        self.explainer = None
        self.features  = None

    def train(self, df: pd.DataFrame) -> dict:
        available = [f for f in FEATURE_COLS if f in df.columns]
        self.features = available
        df_clean = df.dropna(subset=available + ['disengagement'])
        X = df_clean[available].fillna(0)
        y = df_clean['disengagement']
        if len(y.unique()) < 2:
            print('[PredictionAgent] Warning: only one class in labels')
            return {}

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        if SMOTE_AVAILABLE:
            X_tr_b, y_tr_b = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            print(f'[PredictionAgent] SMOTE: {len(y_tr)} → {len(y_tr_b)} training samples')
        else:
            X_tr_b, y_tr_b = X_tr, y_tr
            print('[PredictionAgent] SMOTE unavailable — using scale_pos_weight')

        self.model.fit(X_tr_b, y_tr_b, eval_set=[(X_te, y_te)], verbose=False)

        y_pred  = self.model.predict(X_te)
        y_proba = self.model.predict_proba(X_te)[:, 1]

        metrics = {
            'accuracy':  round(accuracy_score(y_te, y_pred), 4),
            'f1':        round(f1_score(y_te, y_pred, pos_label=1), 4),
            'precision': round(precision_score(y_te, y_pred, pos_label=1, zero_division=0), 4),
            'recall':    round(recall_score(y_te, y_pred, pos_label=1), 4),
            'roc_auc':   round(roc_auc_score(y_te, y_proba), 4),
            'report':    classification_report(y_te, y_pred, target_names=['Healthy','Disengaged']),
        }

        print(f'\n[PredictionAgent] XGBoost hold-out results:')
        print(f'  Accuracy : {metrics["accuracy"]*100:.1f}%')
        print(f'  F1       : {metrics["f1"]*100:.1f}%')
        print(f'  Precision: {metrics["precision"]*100:.1f}%')
        print(f'  Recall   : {metrics["recall"]*100:.1f}%')
        print(f'  ROC-AUC  : {metrics["roc_auc"]:.3f}')
        print(f'\n{metrics["report"]}')

        metrics['cv']        = _run_cross_validation(self.model, X, y)
        metrics['baselines'] = _run_baselines(X_tr, X_te, y_tr, y_te)
        self.explainer       = shap.TreeExplainer(self.model)
        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df        = df.copy()
        available = [f for f in (self.features or FEATURE_COLS) if f in df.columns]
        X         = df[available].fillna(0)
        df['risk_probability'] = self.model.predict_proba(X)[:, 1].round(4)
        df['risk_level'] = pd.cut(df['risk_probability'],
                                   bins=[0, 0.4, 0.65, 0.80, 1.0],
                                   labels=['Low Risk','Moderate Risk','High Risk','Critical Risk'])
        if self.explainer:
            shap_vals = self.explainer.shap_values(X)
            explanations = []
            for row_shap in shap_vals:
                top = np.argsort(np.abs(row_shap))[::-1][:2]
                explanations.append('Driven by: ' + ', '.join(
                    f'{available[j]} ({row_shap[j]:+.2f})' for j in top))
            df['shap_explanation'] = explanations
        return df

    def feature_importance(self) -> pd.DataFrame:
        if not self.features:
            return pd.DataFrame()
        return pd.DataFrame({'feature': self.features,
                              'importance': self.model.feature_importances_}
                             ).sort_values('importance', ascending=False)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.features}, f)
        print(f'[PredictionAgent] Saved → {path}')

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        inst           = cls()
        inst.model     = data['model']
        inst.features  = data['features']
        inst.explainer = shap.TreeExplainer(inst.model)
        print(f'[PredictionAgent] Loaded ← {path}')
        return inst
