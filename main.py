import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# ==========================================
# 0. 한글 폰트 설정
# ==========================================
font_file = "NanumGothic-Regular.ttf"
font_path = os.path.join(os.getcwd(), font_file)

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font=font_name, rc={'axes.unicode_minus': False}, style='whitegrid')
    font_msg = f"✅ 한글 폰트('{font_name}') 적용 완료"
else:
    font_msg = "⚠️ 'NanumGothic-Regular.ttf' 파일 없음"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

# 앱 설정
st.set_page_conf행합니다.")

# ==========================================
# 3. 데이터 나누기 및 모델 설정
# ==========================================
st.header("3. 데이터 나누기 및 모델 설정")

if st.session_state.processed_data is not None:
    X, y = st.session_state.processed_data['X'], st.session_state.processed_data['y']

    ratio = st.radio("데이터 분할 비율", ["7:3", "8:2"], horizontal=True)
    test_size = 0.3 if ratio == "7:3" else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    st.write("**클래스 불균형 해결 방식:** `class_weight='balanced'` (모델 가중치 자동 조정)")

    if st.button("모델 학습 및 평가"):
        def evaluate(model, X_t, y_t, name):
            pred = model.predict(X_t)
            prob = model.predict_proba(X_t)[:, 1]
            st.subheader(f"[{name}] 결과")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("정확도", f"{accuracy_score(y_t, pred):.2f}")
            m2.metric("정밀도", f"{precision_score(y_t, pred):.2f}")
            m3.metric("재현율", f"{recall_score(y_t, pred):.2f}")
            m4.metric("F1-Score", f"{f1_score(y_t, pred):.2f}")

            c1, c2 = st.columns(2)
            with c1:
                sns.heatmap(confusion_matrix(y_t, pred), annot=True, fmt='d', cmap='Blues')
                st.pyplot(plt.gcf())
                plt.clf()
            with c2:
                fpr, tpr, _ = roc_curve(y_t, prob)
                plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
                plt.plot([0, 1], [0, 1], '--')
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()

        # 모델 학습 시 class_weight='balanced' 적용
        st.header("4. 모델 평가(의사결정나무)")
        dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42).fit(X_train, y_train)
        evaluate(dt, X_test, y_test, "의사결정나무")

        st.header("5. 모델 평가(로짓 모델)")
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42).fit(X_train, y_train)
        evaluate(lr, X_test, y_test, "로지스틱 회귀")
else:
    st.info("전처리 실행 버튼을 눌러주세요.")
