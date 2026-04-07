"""
train.py
---------
Runs all 6 agents directly in the Kaggle notebook — no Streamlit needed.
Saves output after each agent so it auto-resumes if the session disconnects.

Usage:
    python train.py

Run via Kaggle notebook cell:
    import subprocess
    subprocess.run(["python", "/kaggle/working/project/train.py"], capture_output=False)
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader               import load_zomato
from agents.nlp_agent          import SentimentAnalyser
from agents.risk_agent         import compute_restaurant_features, compute_health_index
from agents.prediction_agent   import DisengagementPredictor, build_disengagement_label
from agents.strategy_agent     import apply_strategies, build_action_list
from agents.verification_agent import Verifier
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = '/kaggle/input/datasets/himanshupoddar/zomato-bangalore-restaurants/zomato.csv'
RECENT_PATH = '/kaggle/input/datasets/teshavsharma/zomato-restaurant-data/zomato.csv'
SAVE_DIR    = '/kaggle/working/saved_results'
NROWS       = 3000   # rows from historical dataset — 3000 runs in ~20 mins on T4 GPU

os.makedirs(SAVE_DIR, exist_ok=True)


def done(filename: str) -> bool:
    """Returns True if file already exists — agent will be skipped."""
    path = f'{SAVE_DIR}/{filename}'
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        print(f'  ⏩ Skipping — {filename} already saved ({kb:.0f} KB)')
        return True
    return False


print('=' * 55)
print('Restaurant Disengagement Risk — Training Pipeline')
print('=' * 55)

# ── Agent 1: Data Loading ─────────────────────────────────────────────────────
print('\n[1/6] Data Loader')
if not done('reviews_raw.csv'):
    reviews_df = load_zomato(DATA_PATH, nrows=NROWS, recent_path=RECENT_PATH)
    reviews_df.to_csv(f'{SAVE_DIR}/reviews_raw.csv', index=False)
    print(f'  ✅ {len(reviews_df):,} reviews saved')
else:
    reviews_df = pd.read_csv(f'{SAVE_DIR}/reviews_raw.csv')
    print(f'  ✅ Loaded {len(reviews_df):,} reviews from file')

# ── Agent 2: NLP / ABSA ───────────────────────────────────────────────────────
print('\n[2/6] NLP Sentiment Analysis  (this takes ~15–20 mins on GPU)')
if not done('reviews_results.csv') or not done('sentiment.csv'):
    analyser     = SentimentAnalyser()
    reviews_df   = analyser.analyse_reviews(reviews_df)
    sentiment_df = analyser.aggregate_to_restaurant(reviews_df)
    reviews_df.to_csv(f'{SAVE_DIR}/reviews_results.csv',   index=False)
    sentiment_df.to_csv(f'{SAVE_DIR}/sentiment.csv',        index=False)
    print('  ✅ Sentiment scores saved')
else:
    reviews_df   = pd.read_csv(f'{SAVE_DIR}/reviews_results.csv')
    sentiment_df = pd.read_csv(f'{SAVE_DIR}/sentiment.csv')
    print('  ✅ Loaded from file')

# ── Agent 3: Restaurant Health Index ─────────────────────────────────────────
print('\n[3/6] Restaurant Health Index')
if not done('restaurant_pre_pred.csv'):
    rest_features = compute_restaurant_features(reviews_df)
    restaurant_df = compute_health_index(rest_features, sentiment_df)
    restaurant_df.to_csv(f'{SAVE_DIR}/restaurant_pre_pred.csv', index=False)
    print('  ✅ Health index saved')
else:
    restaurant_df = pd.read_csv(f'{SAVE_DIR}/restaurant_pre_pred.csv')
    print('  ✅ Loaded from file')

# ── Agent 4: XGBoost + SMOTE + Cross-Validation ───────────────────────────────
print('\n[4/6] XGBoost Prediction (SMOTE + 5-fold CV + baseline comparison)')
if not done('model.pkl') or not done('restaurant_results.csv'):
    restaurant_df = build_disengagement_label(restaurant_df)
    predictor     = DisengagementPredictor()
    metrics       = predictor.train(restaurant_df)
    restaurant_df = predictor.predict(restaurant_df)
    predictor.save(f'{SAVE_DIR}/model.pkl')
    restaurant_df.to_csv(f'{SAVE_DIR}/restaurant_results.csv', index=False)
    with open(f'{SAVE_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, default=str)
    print(f'  ✅ Accuracy:{metrics["accuracy"]*100:.1f}% | '
          f'F1:{metrics["f1"]*100:.1f}% | '
          f'Precision:{metrics["precision"]*100:.1f}% | '
          f'Recall:{metrics["recall"]*100:.1f}% | '
          f'AUC:{metrics["roc_auc"]:.3f}')
else:
    restaurant_df = pd.read_csv(f'{SAVE_DIR}/restaurant_results.csv')
    with open(f'{SAVE_DIR}/metrics.json') as f:
        metrics = json.load(f)
    print(f'  ✅ Loaded — Accuracy: {metrics["accuracy"]*100:.1f}%')

# ── Agent 5: Strategy Generation ─────────────────────────────────────────────
print('\n[5/6] Retention Strategy Generation')
if not done('action_list.csv'):
    restaurant_df = apply_strategies(restaurant_df)
    action_df     = build_action_list(restaurant_df, top_n=20)
    action_df.to_csv(f'{SAVE_DIR}/action_list.csv', index=False)
    restaurant_df.to_csv(f'{SAVE_DIR}/restaurant_results.csv', index=False)
    print(f'  ✅ Strategies for {len(action_df)} at-risk restaurants')
else:
    action_df = pd.read_csv(f'{SAVE_DIR}/action_list.csv')
    print('  ✅ Loaded from file')

# ── Agent 6: Verification ─────────────────────────────────────────────────────
print('\n[6/6] Verification')
verifier     = Verifier()
verification = verifier.run(reviews_df, restaurant_df, action_df)
print(f'  ✅ {verification["summary"]}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('TRAINING COMPLETE')
print('=' * 55)
print(f'Saved files in {SAVE_DIR}:')
for f in sorted(os.listdir(SAVE_DIR)):
    kb = os.path.getsize(f'{SAVE_DIR}/{f}') / 1024
    print(f'  {f:42s} {kb:>8.0f} KB')
print('\nNext step: open the dashboard and switch to Demo Mode.')
