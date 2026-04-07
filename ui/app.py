"""
ui/app.py
----------
Restaurant Disengagement Risk Analysis — Streamlit Dashboard
MDA681 | Sweena Jennifer Sandra J (2482434) | CHRIST University

Three modes:
  📊 Demo Mode    — loads saved results instantly (use on demo day)
  🔴 Live Reviews — SerpApi live scoring against saved model
  ⚙️ Train Mode   — runs full pipeline (use train.py in notebook instead)
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.path.exists('/kaggle/input'):
    DEFAULT_DATA  = '/kaggle/input/datasets/himanshupoddar/zomato-bangalore-restaurants/zomato.csv'
    RECENT_DATA   = '/kaggle/input/datasets/teshavsharma/zomato-restaurant-data/zomato.csv'
    SAVE_DIR      = '/kaggle/working/saved_results'
else:
    DEFAULT_DATA  = '/content/restaurant_risk/data/zomato_bangalore_historical.csv'
    RECENT_DATA   = '/content/restaurant_risk/data/zomato_recent_2024_2025.csv'
    SAVE_DIR      = '/content/drive/MyDrive/MDA681/saved_results'

MODEL_PATH = os.path.join(SAVE_DIR, 'model.pkl')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Restaurant Risk Analysis', page_icon='🍽️', layout='wide')

# ── Session state ─────────────────────────────────────────────────────────────
for key in ['done','reviews_df','restaurant_df','action_df',
            'metrics','verification','overrides','analyser','predictor']:
    if key not in st.session_state:
        st.session_state[key] = (False if key == 'done'
                                  else {} if key == 'overrides' else None)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image('https://upload.wikimedia.org/wikipedia/en/thumb/7/72/'
                 'Christ_University_logo.png/200px-Christ_University_logo.png', width=100)
    except Exception:
        pass
    st.title('🍽️ Restaurant Risk MAS')
    st.caption('CHRIST University | MDA681')
    st.markdown('---')
    mode = st.radio('Mode', [
        '📊 Demo Mode',
        '🔴 Live Reviews (SerpApi)',
        '⚙️ Train Mode',
    ], index=0)
    st.markdown('---')
    if '⚙️' in mode:
        DATA_PATH      = st.text_input('Dataset path', value=DEFAULT_DATA)
        NROWS          = st.number_input('Rows (0=all)', 0, 200000, 3000, 1000)
    RISK_THRESHOLD = st.slider('Risk threshold', 0.3, 0.9, 0.5, 0.05)
    st.markdown('---')
    st.markdown('**Agents**')
    st.markdown('1️⃣ Data Loader\n\n2️⃣ NLP/ABSA\n\n3️⃣ Risk Scoring\n\n'
                '4️⃣ Prediction\n\n5️⃣ Strategy\n\n6️⃣ Verification ✅')


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE
# ─────────────────────────────────────────────────────────────────────────────
if '📊' in mode:
    st.title('🍽️ Restaurant Disengagement Risk Analysis')
    st.caption('CHRIST (Deemed to be University) | MDA681 | Bangalore Restaurant Study')

    if not st.session_state.done:
        try:
            from agents.prediction_agent import DisengagementPredictor
            st.session_state.restaurant_df = pd.read_csv(f'{SAVE_DIR}/restaurant_results.csv')
            st.session_state.reviews_df    = pd.read_csv(f'{SAVE_DIR}/reviews_results.csv')
            st.session_state.action_df     = pd.read_csv(f'{SAVE_DIR}/action_list.csv')
            with open(f'{SAVE_DIR}/metrics.json') as f:
                st.session_state.metrics = json.load(f)
            if os.path.exists(MODEL_PATH):
                st.session_state.predictor = DisengagementPredictor.load(MODEL_PATH)
            st.session_state.done         = True
            st.session_state.verification = {
                'passed': True, 'issues': [], 'warnings': [],
                'flagged_rows': pd.DataFrame(),
                'summary': '✅ Results loaded from saved training run.'
            }
            st.success('✅ Results loaded instantly — ready to present!')
        except FileNotFoundError:
            st.warning('No saved results found.')
            st.info('Run `train.py` in your Kaggle notebook first, then come back to Demo Mode.')
            st.stop()

    if st.session_state.done and st.session_state.restaurant_df is not None:
        df        = st.session_state.restaurant_df
        rev_df    = st.session_state.reviews_df
        action_df = st.session_state.action_df
        metrics   = st.session_state.metrics
        verif     = st.session_state.verification

        # Verification banner
        st.markdown('---')
        st.subheader('🔍 Agent 6: Verification')
        for issue in verif.get('issues', []):
            st.error(f'🔴 {issue["check"]}: {issue["detail"]}')
        for warn in verif.get('warnings', []):
            if str(warn.get('detail','')).startswith('✅'):
                st.success(f'{warn["check"]}: {warn["detail"]}')
            else:
                st.warning(f'🟡 {warn["check"]}: {warn["detail"]}')

        # KPI row
        st.markdown('---')
        k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
        n_risk = int((df['risk_probability'] >= RISK_THRESHOLD).sum()) \
                 if 'risk_probability' in df.columns else 0
        k1.metric('Restaurants',  f'{len(df):,}')
        k2.metric('Reviews',      f'{len(rev_df):,}')
        k3.metric('At-Risk',      n_risk)
        k4.metric('Accuracy',     f'{metrics.get("accuracy",0)*100:.1f}%')
        k5.metric('F1 Score',     f'{metrics.get("f1",0)*100:.1f}%')
        k6.metric('Precision',    f'{metrics.get("precision",0)*100:.1f}%')
        k7.metric('Recall',       f'{metrics.get("recall",0)*100:.1f}%')

        # ── Model Evaluation Table ────────────────────────────────────────────
        st.markdown('---')
        st.subheader('📋 Model Evaluation — XGBoost vs Baselines')

        eval_rows = []
        # XGBoost hold-out
        eval_rows.append({
            'Model':     '⭐ XGBoost (proposed)',
            'Accuracy':  f'{metrics.get("accuracy",0)*100:.1f}%',
            'F1 Score':  f'{metrics.get("f1",0)*100:.1f}%',
            'Precision': f'{metrics.get("precision",0)*100:.1f}%',
            'Recall':    f'{metrics.get("recall",0)*100:.1f}%',
            'ROC-AUC':   f'{metrics.get("roc_auc",0):.3f}',
            'Note':      'Hold-out test set (20%)',
        })
        # Cross-validation
        cv = metrics.get('cv', {})
        if cv:
            eval_rows.append({
                'Model':     '⭐ XGBoost (5-fold CV)',
                'Accuracy':  f'{cv.get("accuracy_mean",0)*100:.1f}% ± {cv.get("accuracy_std",0)*100:.1f}%',
                'F1 Score':  f'{cv.get("f1_mean",0)*100:.1f}% ± {cv.get("f1_std",0)*100:.1f}%',
                'Precision': f'{cv.get("precision_mean",0)*100:.1f}% ± {cv.get("precision_std",0)*100:.1f}%',
                'Recall':    f'{cv.get("recall_mean",0)*100:.1f}% ± {cv.get("recall_std",0)*100:.1f}%',
                'ROC-AUC':   f'{cv.get("roc_auc_mean",0):.3f} ± {cv.get("roc_auc_std",0):.3f}',
                'Note':      'Mean ± Std across 5 folds',
            })
        # Baselines
        for name, bm in metrics.get('baselines', {}).items():
            eval_rows.append({
                'Model':     name,
                'Accuracy':  f'{bm.get("accuracy",0)*100:.1f}%',
                'F1 Score':  f'{bm.get("f1",0)*100:.1f}%',
                'Precision': f'{bm.get("precision",0)*100:.1f}%',
                'Recall':    f'{bm.get("recall",0)*100:.1f}%',
                'ROC-AUC':   '—',
                'Note':      'Baseline comparison',
            })

        if eval_rows:
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

        # Charts
        st.markdown('---')
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('📊 Health Segments')
            if 'segment' in df.columns:
                seg = df['segment'].value_counts().reset_index()
                seg.columns = ['Segment','Count']
                fig1 = px.pie(seg, names='Segment', values='Count', color='Segment',
                              color_discrete_map={'Thriving':'#22C55E','Stable':'#3B82F6',
                                                  'At-Risk':'#F59E0B','Critical':'#EF4444'})
                st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.subheader('🔥 Risk Distribution')
            if 'risk_probability' in df.columns:
                fig2 = px.histogram(df, x='risk_probability', nbins=25,
                                    color_discrete_sequence=['#EF4444'])
                fig2.add_vline(x=RISK_THRESHOLD, line_dash='dash', line_color='navy',
                               annotation_text=f'Threshold {RISK_THRESHOLD}')
                st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader('💬 Aspect Sentiment by Segment')
            asp = ['food_sentiment','service_sentiment','ambiance_sentiment']
            if all(c in df.columns for c in asp) and 'segment' in df.columns:
                melt = df.groupby('segment')[asp].mean().reset_index().melt(
                    id_vars='segment', value_vars=asp, var_name='Aspect', value_name='Score')
                melt['Aspect'] = melt['Aspect'].str.replace('_sentiment','').str.title()
                fig3 = px.bar(melt, x='segment', y='Score', color='Aspect', barmode='group',
                              color_discrete_sequence=['#E67E22','#3498DB','#9B59B6'])
                st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.subheader('📉 Sentiment Trend by Area')
            if 'location' in df.columns and 'sentiment_trend' in df.columns:
                at = df.groupby('location')['sentiment_trend'].mean().reset_index()
                at = at.sort_values('sentiment_trend').head(15)
                fig4 = px.bar(at, x='location', y='sentiment_trend',
                              color='sentiment_trend', color_continuous_scale='RdYlGn',
                              range_color=[-0.3, 0.3])
                fig4.add_hline(y=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig4, use_container_width=True)

        # EDA
        st.markdown('---')
        st.subheader('📊 EDA Findings')
        e1, e2 = st.columns(2)
        with e1:
            st.markdown('**Cost by Area (Inflation Signal)**')
            if 'price_for_two' in df.columns and 'location' in df.columns:
                top = df['location'].value_counts().head(10).index
                fig5 = px.box(df[df['location'].isin(top) & df['price_for_two'].notna()],
                              x='location', y='price_for_two', color='location',
                              labels={'price_for_two':'Price for Two (₹)'})
                fig5.update_layout(showlegend=False)
                st.plotly_chart(fig5, use_container_width=True)
        with e2:
            st.markdown('**Dominant Complaints**')
            if 'dominant_complaint' in df.columns:
                comp = df['dominant_complaint'].value_counts().reset_index()
                comp.columns = ['Complaint','Count']
                fig6 = px.bar(comp, x='Complaint', y='Count',
                              color='Count', color_continuous_scale='Reds')
                st.plotly_chart(fig6, use_container_width=True)

        # Feature importance
        st.markdown('---')
        st.subheader('🔬 Feature Importance (SHAP)')
        predictor = st.session_state.get('predictor')
        if predictor:
            fi = predictor.feature_importance()
            if not fi.empty:
                fig7 = px.bar(fi.head(10), x='importance', y='feature', orientation='h',
                              color='importance', color_continuous_scale='Teal')
                fig7.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig7, use_container_width=True)

        # Human review
        st.markdown('---')
        st.subheader('🧑‍💼 Human Review — Flagged Restaurants')
        flagged = verif.get('flagged_rows', pd.DataFrame())
        if not flagged.empty:
            st.warning(f'{len(flagged)} restaurants flagged. Edit strategy if needed.')
            edit_cols = [c for c in ['restaurant_name','flag_reason','segment',
                                      'dominant_complaint','strategy'] if c in flagged.columns]
            edited = st.data_editor(flagged[edit_cols], use_container_width=True,
                                     num_rows='fixed',
                                     column_config={'strategy': st.column_config.TextColumn(
                                         'Strategy (editable)', width='large')},
                                     disabled=[c for c in edit_cols if c != 'strategy'],
                                     key='human_editor')
            if st.button('✅ Apply Overrides'):
                overrides = {}
                if 'restaurant_name' in edited.columns:
                    orig = flagged.set_index('restaurant_name')['strategy']
                    for _, row in edited.iterrows():
                        n, v = row.get('restaurant_name',''), row.get('strategy','')
                        if v != orig.get(n,'') and v:
                            overrides[n] = v
                st.session_state.overrides = overrides
                st.success(f'Applied {len(overrides)} override(s).')
        else:
            st.success('✅ No rows flagged — all outputs passed quality checks.')

        # Action list
        st.markdown('---')
        st.subheader('🎯 Priority Action List')
        disp = action_df.copy()
        for n, v in st.session_state.get('overrides', {}).items():
            if 'restaurant_name' in disp.columns:
                disp.loc[disp['restaurant_name']==n, 'strategy'] = f'[HUMAN OVERRIDE] {v}'
        st.dataframe(disp, use_container_width=True, height=450)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button('📥 Action List (CSV)', data=disp.to_csv(index=False),
                               file_name='action_list.csv', mime='text/csv')
        with c2:
            st.download_button('📥 Full Results (CSV)', data=df.to_csv(index=False),
                               file_name='full_results.csv', mime='text/csv')


# ─────────────────────────────────────────────────────────────────────────────
# LIVE REVIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif '🔴' in mode:
    st.title('🔴 Live Review Scoring')
    st.caption('Type any Bangalore restaurant → fetches real Google Maps reviews → scores instantly')
    st.info('Setup: serpapi.com → sign up free → paste key in agents/live_reviews_agent.py')

    if st.session_state.predictor is None and os.path.exists(MODEL_PATH):
        from agents.prediction_agent import DisengagementPredictor
        st.session_state.predictor = DisengagementPredictor.load(MODEL_PATH)

    if st.session_state.analyser is None:
        from agents.nlp_agent import SentimentAnalyser
        with st.spinner('Loading NLP model...'):
            st.session_state.analyser = SentimentAnalyser()

    col1, col2 = st.columns([3,1])
    with col1:
        rest_input = st.text_input('Restaurant name', placeholder='e.g. Onesta Koramangala')
    with col2:
        num_reviews = st.number_input('Max reviews', 5, 50, 20)

    if st.button('🔍 Fetch & Score', type='primary'):
        if not rest_input.strip():
            st.warning('Enter a restaurant name.')
        else:
            with st.spinner(f'Fetching live reviews for {rest_input}...'):
                from agents.live_reviews_agent import fetch_live_reviews, score_live_reviews
                live_df = fetch_live_reviews(rest_input, num_reviews=num_reviews)
                if live_df.empty:
                    st.error('No reviews fetched. Check your SerpApi key.')
                else:
                    result = score_live_reviews(live_df, st.session_state.analyser,
                                                 st.session_state.predictor)
                    st.subheader(f'📍 {result["restaurant_name"]}')

                    # Metrics
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric('Snippets Fetched', result['total_reviews'])
                    m2.metric('Avg Rating',        result.get('avg_rating') or 'N/A')
                    m3.metric('Risk Level',        f'{result["risk_color"]} {result["risk_level"]}')
                    m4.metric('Churn Intent',      f'{result["churn_intent_count"]} mentions')

                    # Sentiment chart
                    fig = px.bar(pd.DataFrame({
                        'Aspect': ['Food','Service','Ambiance'],
                        'Score':  [result['food_sentiment'],
                                   result['service_sentiment'],
                                   result['ambiance_sentiment']]
                    }), x='Aspect', y='Score', color='Score',
                        color_continuous_scale='RdYlGn', range_color=[-1,1],
                        title='Live Sentiment Scores')
                    fig.add_hline(y=0, line_dash='dash', line_color='gray')
                    st.plotly_chart(fig, use_container_width=True)

                    # Strategy recommendation
                    from agents.strategy_agent import generate_strategy
                    dominant   = result.get('dominant_complaint', 'None')
                    risk_level = result.get('risk_level', 'Low Risk')
                    if 'Critical' in risk_level or 'High' in risk_level:
                        segment = 'Critical'
                    elif 'Moderate' in risk_level:
                        segment = 'At-Risk'
                    else:
                        segment = 'Stable'
                    strategy = generate_strategy(segment, dominant)

                    st.markdown('---')
                    st.subheader('🎯 Recommended Strategy')
                    st.info(
                        '**Segment:** ' + segment +
                        ' | **Dominant Complaint:** ' + dominant +
                        chr(10) + chr(10) + strategy
                    )

                    # Raw snippets table
                    st.markdown('---')
                    st.subheader('📋 Review Snippets Fetched')
                    show = [c for c in ['reviewer_name','review_rating','review_text',
                                         'overall_sentiment','dominant_complaint']
                            if c in result['reviews'].columns]
                    st.dataframe(result['reviews'][show], use_container_width=True)

    if not os.path.exists(MODEL_PATH):
        st.warning('No saved model found. Run train.py first.')


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────────────────────────────────────────
elif '⚙️' in mode:
    st.title('⚙️ Train Mode')
    st.warning(
        '**Recommended:** Run `train.py` directly in your Kaggle notebook cell '
        'instead of here. It is faster, shows live output, and auto-resumes if '
        'the session disconnects.\n\n'
        'Use this mode only if you want to run the pipeline through Streamlit.'
    )

    if st.button('▶️ Run Full Pipeline & Save', type='primary'):
        from data.loader               import load_zomato
        from agents.nlp_agent          import SentimentAnalyser
        from agents.risk_agent         import compute_restaurant_features, compute_health_index
        from agents.prediction_agent   import DisengagementPredictor, build_disengagement_label
        from agents.strategy_agent     import apply_strategies, build_action_list
        from agents.verification_agent import Verifier

        nrows = int(NROWS) if NROWS > 0 else None
        prog  = st.progress(0, text='Starting...')
        try:
            prog.progress(5,  text='Agent 1/6: Loading data...')
            reviews_df = load_zomato(DATA_PATH, nrows=nrows, recent_path=RECENT_DATA)
            st.success(f'✅ Agent 1 — {len(reviews_df):,} reviews')

            prog.progress(15, text='Agent 2/6: DistilBERT sentiment...')
            analyser     = SentimentAnalyser()
            reviews_df   = analyser.analyse_reviews(reviews_df)
            sentiment_df = analyser.aggregate_to_restaurant(reviews_df)
            st.session_state.analyser = analyser
            st.success('✅ Agent 2 done')

            prog.progress(60, text='Agent 3/6: Health Index...')
            rest_features = compute_restaurant_features(reviews_df)
            restaurant_df = compute_health_index(rest_features, sentiment_df)
            st.success(f'✅ Agent 3 — {restaurant_df["segment"].value_counts().to_dict()}')

            prog.progress(75, text='Agent 4/6: XGBoost + SMOTE + CV...')
            restaurant_df = build_disengagement_label(restaurant_df)
            predictor     = DisengagementPredictor()
            metrics       = predictor.train(restaurant_df)
            restaurant_df = predictor.predict(restaurant_df)
            st.session_state.predictor = predictor
            st.success(f'✅ Agent 4 — Acc:{metrics.get("accuracy",0)*100:.1f}% '
                       f'F1:{metrics.get("f1",0)*100:.1f}% '
                       f'AUC:{metrics.get("roc_auc",0):.3f}')

            prog.progress(88, text='Agent 5/6: Strategies...')
            restaurant_df = apply_strategies(restaurant_df)
            action_df     = build_action_list(restaurant_df, top_n=20)
            st.success(f'✅ Agent 5 — {len(action_df)} at-risk restaurants')

            prog.progress(95, text='Agent 6/6: Verification...')
            verification = Verifier().run(reviews_df, restaurant_df, action_df)
            st.success(f'✅ Agent 6 — {verification["summary"]}')
            prog.progress(98, text='Saving...')

            restaurant_df.to_csv(f'{SAVE_DIR}/restaurant_results.csv', index=False)
            reviews_df.to_csv(f'{SAVE_DIR}/reviews_results.csv', index=False)
            action_df.to_csv(f'{SAVE_DIR}/action_list.csv', index=False)
            with open(f'{SAVE_DIR}/metrics.json', 'w') as f:
                json.dump(metrics, f, default=str)
            predictor.save(MODEL_PATH)

            prog.progress(100, text='Done!')
            st.session_state.update({
                'done': True, 'reviews_df': reviews_df,
                'restaurant_df': restaurant_df, 'action_df': action_df,
                'metrics': metrics, 'verification': verification
            })
            st.balloons()
            st.success(f'💾 All results saved to {SAVE_DIR}')
            st.info('Switch to 📊 Demo Mode to present results.')
        except Exception as e:
            st.error(f'Pipeline error: {e}')
            st.exception(e)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.caption('MDA681 | Sweena Jennifer Sandra J (2482434) | '
           'CHRIST (Deemed to be University) | Guide: Dr. Dalvin Vinoth Kumar')
