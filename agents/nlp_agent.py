"""
agents/nlp_agent.py
"""
import re
import torch
import gc
import pandas as pd
import numpy as np
from transformers import pipeline

ASPECT_KEYWORDS = {
    'Food': [
        'food', 'taste', 'flavour', 'flavor', 'dish', 'meal', 'menu',
        'biryani', 'pizza', 'burger', 'dosa', 'spicy', 'fresh', 'stale',
        'chef', 'portion', 'curry', 'roti', 'rice', 'thali', 'masala',
        'paneer', 'chicken', 'sweet', 'dessert', 'starter', 'buffet'
    ],
    'Service': [
        'service', 'staff', 'waiter', 'delivery', 'speed', 'slow', 'fast',
        'rude', 'helpful', 'attentive', 'wait', 'delay', 'manager',
        'server', 'order', 'response', 'courteous', 'behaviour', 'behavior'
    ],
    'Ambiance': [
        'ambiance', 'ambience', 'atmosphere', 'decor', 'noise', 'clean',
        'dirty', 'cozy', 'crowded', 'music', 'lighting', 'seating',
        'interior', 'outdoor', 'parking', 'washroom', 'table', 'view'
    ],
}

COMPLAINT_THRESHOLD = -0.20


class SentimentAnalyser:

    def __init__(self):
        print('[NLPAgent] Loading DistilBERT...')
        self.pipe = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=0 if torch.cuda.is_available() else -1,
            batch_size=64,
            truncation=True,
            max_length=128,
        )
        print(f'[NLPAgent] Model loaded on {"GPU" if torch.cuda.is_available() else "CPU"} ✓')

    def _bucket_sentences(self, text: str) -> dict:
        sentences = re.split(r'[.!?,;]', text.lower())
        buckets   = {asp: [] for asp in ASPECT_KEYWORDS}
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(kw in sent for kw in keywords):
                    buckets[aspect].append(sent)
        return buckets

    def _batch_score_all(self, sentences: list) -> list:
        if not sentences:
            return []
        results    = []
        batch_size = 64
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            out   = self.pipe(batch)
            for r in out:
                results.append(r['score'] if r['label'] == 'POSITIVE' else -r['score'])
            if i % 20000 == 0 and i > 0:
                print(f'    Scored {i:,}/{len(sentences):,}...')
                torch.cuda.empty_cache()
                gc.collect()
        return results

    def analyse_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df['review_text'].fillna('').tolist()
        print(f'[NLPAgent] Bucketing {len(texts):,} reviews...')

        food_sentences, svc_sentences, amb_sentences = [], [], []
        food_counts,    svc_counts,    amb_counts    = [], [], []

        for text in texts:
            buckets = self._bucket_sentences(str(text))
            food_sentences.extend(buckets['Food'])
            svc_sentences.extend(buckets['Service'])
            amb_sentences.extend(buckets['Ambiance'])
            food_counts.append(len(buckets['Food']))
            svc_counts.append(len(buckets['Service']))
            amb_counts.append(len(buckets['Ambiance']))

        print(f'[NLPAgent] Scoring {len(food_sentences):,} food sentences...')
        food_flat = self._batch_score_all(food_sentences)
        torch.cuda.empty_cache()
        gc.collect()

        print(f'[NLPAgent] Scoring {len(svc_sentences):,} service sentences...')
        svc_flat = self._batch_score_all(svc_sentences)
        torch.cuda.empty_cache()
        gc.collect()

        print(f'[NLPAgent] Scoring {len(amb_sentences):,} ambiance sentences...')
        amb_flat = self._batch_score_all(amb_sentences)
        torch.cuda.empty_cache()
        gc.collect()

        def split_by_counts(flat, counts):
            result, idx = [], 0
            for c in counts:
                chunk = flat[idx:idx + c]
                result.append(round(float(sum(chunk) / len(chunk)), 4) if chunk else 0.0)
                idx += c
            return result

        df = df.copy()
        df['food_sentiment']     = split_by_counts(food_flat, food_counts)
        df['service_sentiment']  = split_by_counts(svc_flat,  svc_counts)
        df['ambiance_sentiment'] = split_by_counts(amb_flat,  amb_counts)
        df['overall_sentiment']  = df[
            ['food_sentiment', 'service_sentiment', 'ambiance_sentiment']
        ].mean(axis=1).round(4)

        asp_cols    = ['food_sentiment', 'service_sentiment', 'ambiance_sentiment']
        worst_score = df[asp_cols].min(axis=1)
        worst_asp   = df[asp_cols].idxmin(axis=1).str.replace('_sentiment', '').str.title()
        df['dominant_complaint'] = worst_asp.where(
            worst_score < COMPLAINT_THRESHOLD, other='None'
        )

        print('[NLPAgent] Sentiment done ✓')
        return df

    def aggregate_to_restaurant(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['is_recent'] = df.groupby('restaurant_name')['review_index'].transform(
            lambda x: x >= x.quantile(0.80)
        )
        recent_sentiment = df[df['is_recent']].groupby('restaurant_name')[
            'overall_sentiment'
        ].mean().rename('recent_sentiment')

        agg = df.groupby('restaurant_name').agg(
            food_sentiment     = ('food_sentiment',     'mean'),
            service_sentiment  = ('service_sentiment',  'mean'),
            ambiance_sentiment = ('ambiance_sentiment', 'mean'),
            overall_sentiment  = ('overall_sentiment',  'mean'),
            dominant_complaint = ('dominant_complaint',
                                  lambda x: x[x != 'None'].mode()[0]
                                  if len(x[x != 'None']) > 0 else 'None'),
            churn_rate         = ('churn', 'mean'),
        ).reset_index()

        agg = agg.merge(recent_sentiment, on='restaurant_name', how='left')
        agg['sentiment_trend'] = (
            agg['recent_sentiment'].fillna(agg['overall_sentiment'])
            - agg['overall_sentiment']
        ).round(4)

        print(f'[NLPAgent] Aggregated {len(agg)} restaurants ✓')
        return agg
