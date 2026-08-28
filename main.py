import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

def password_check():
    if st.session_state.get("authenticated"):
        return

    password = st.text_input("비밀번호를 입력하세요", type="password")

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")

    st.stop()

password_check()

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
st.set_page_config(page_title="고객 이탈 예측 분석", layout="wide")
st.sidebar.info(font_msg)
st.title("📊 고객 이탈 관리 데이터 분석 ")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Churn_management.csv")

try:
    df = load_data()
    st.success("데이터 로드 완료!")
except:
    st.error("'Churn_management.csv' 파일이 필요합니다.")
    st.stop()

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# ==========================================
# 1. 데이터 분석 및 시각화
# ==========================================
st.header("1. 데이터 분석 및 시각화")
col_dist, col_custom = st.columns([1, 2])

with col_dist:
    st.subheader("종속 변수(Exited) 분포")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Exited', data=df, ax=ax, palette='viridis')
    ax.set_title("고객 이탈 여부 분포")
    st.pyplot(fig)

with col_custom:
    st.subheader("변수별 시각화")
    c1, c2, c3 = st.columns(3)
    p_type = c1.selectbox("그래프 유형", ["히스토그램", "박스 플롯", "산점도", "막대 차트"])
    x_v = c2.selectbox("X축 변수", df.columns)
    y_v = c3.selectbox("Y축 변수", [None] + list(df.columns))
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if p_type == "히스토그램": sns.histplot(data=df, x=x_v, kde=True, ax=ax2)
    elif p_type == "박스 플롯": sns.boxplot(data=df, x=x_v, y=y_v, ax=ax2)
    elif p_type == "산점도" and y_v: sns.scatterplot(data=df, x=x_v, y=y_v, hue='Exited', ax=ax2)
    elif p_type == "막대 차트": sns.countplot(data=df, x=x_v, ax=ax2) if not y_v else sns.barplot(data=df, x=x_v, y=y_v, ax=ax2)
    st.pyplot(fig2)

# ==========================================
# 2. 데이터 전처리
# ==========================================
st.header("2. 데이터 전처리")

if st.button("데이터 전처리 실행"):
    with st.spinner("데이터 정제 및 변수 선택 중..."):
        df_p = df.copy()
        # 1. 식별자 제거
        df_p = df_p.drop(columns=[c for c in ['id', 'CustomerId', 'Surname'] if c in df_p.columns])
        
        # 2. 결측치 및 이상치
        nums = df_p.select_dtypes(include=[np.number]).columns.drop('Exited', errors='ignore')
        cats = df_p.select_dtypes(include=['object']).columns
        df_p[nums] = df_p[nums].fillna(df_p[nums].mean())
        for c in nums:
            q1, q3 = df_p[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_p[c] = np.clip(df_p[c], q1 - 1.5*iqr, q3 + 1.5*iqr)

        # 3. 인코딩 및 스케일링
        df_p = pd.get_dummies(df_p, columns=cats, drop_first=True)
        sc = StandardScaler()
        scale_cols = [c for c in df_p.columns if c != 'Exited']
        df_p[scale_cols] = sc.fit_transform(df_p[scale_cols])

        X, y = df_p.drop('Exited', axis=1), df_p['Exited']

        # 4. 변수 선택 (Stepwise) - SMOTE 없이 원본 비율 유지하여 선택
        sample_X = X.sample(n=min(5000, len(X)), random_state=42)
        sample_y = y.loc[sample_X.index]
        sfs = SequentialFeatureSelector(LogisticRegression(max_iter=100), n_features_to_select='auto').fit(sample_X, sample_y)
        X = X[X.columns[sfs.get_support()]]
        
        st.session_state.processed_data = {'X': X, 'y': y}
        st.success(f"✅ 전처리 완료!")

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
