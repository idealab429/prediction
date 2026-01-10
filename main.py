import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# ==========================================
# 0. 한글 폰트 완벽 설정 (최우선 실행)
# ==========================================
font_file = "NanumGothic-Regular.ttf"
font_path = os.path.join(os.getcwd(), font_file)

if os.path.exists(font_path):
    # 1. Matplotlib에 폰트 파일 직접 등록
    fm.fontManager.addfont(font_path)
    # 2. 등록된 폰트의 정확한 이름 가져오기
    font_name = fm.FontProperties(fname=font_path).get_name()
    # 3. 모든 설정에 해당 폰트 적용
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    # 4. Seaborn에도 폰트 적용
    sns.set(font=font_name, rc={'axes.unicode_minus': False}, style='whitegrid')
    font_status = f"✅ 한글 폰트('{font_name}') 적용 완료"
else:
    font_status = "⚠️ 폰트 파일을 찾을 수 없습니다. (NanumGothic-Regular.ttf 확인 필요)"

# 나머지 라이브러리 임포트
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False

# 앱 설정
st.set_page_config(page_title="고객 이탈 예측 앱", layout="wide")
st.sidebar.info(font_status)
st.title("📊 고객 이탈 관리 데이터 분석 앱")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Churn_management.csv")

try:
    df = load_data()
except:
    st.error("데이터 파일을 찾을 수 없습니다.")
    st.stop()

# 세션 상태 초기화
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# ==========================================
# 1. 데이터 분석 및 시각화
# ==========================================
st.header("1. 데이터 분석 및 시각화")

col_dist, col_custom = st.columns([1, 2])

with col_dist:
    st.subheader("이탈 여부 분포")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Exited', data=df, ax=ax, palette='viridis')
    ax.set_title("고객 이탈 여부 분포 (0:유지, 1:이탈)")
    ax.set_xlabel("이탈 여부")
    ax.set_ylabel("고객 수")
    st.pyplot(fig)

with col_custom:
    st.subheader("변수별 시각화")
    c1, c2, c3 = st.columns(3)
    p_type = c1.selectbox("그래프 유형", ["히스토그램", "박스 플롯", "산점도", "막대 차트"])
    x_v = c2.selectbox("X축 변수", df.columns)
    y_v = c3.selectbox("Y축 변수 (선택)", [None] + list(df.columns))

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if p_type == "히스토그램":
        sns.histplot(data=df, x=x_v, kde=True, ax=ax2)
    elif p_type == "박스 플롯":
        sns.boxplot(data=df, x=x_v, y=y_v, ax=ax2)
    elif p_type == "산점도" and y_v:
        sns.scatterplot(data=df, x=x_v, y=y_v, hue='Exited', ax=ax2)
    elif p_type == "막대 차트":
        if y_v: sns.barplot(data=df, x=x_v, y=y_v, ax=ax2)
        else: sns.countplot(data=df, x=x_v, ax=ax2)
    
    ax2.set_title(f"{x_v} {p_type}")
    st.pyplot(fig2)

# ==========================================
# 2. 데이터 전처리 (속도 최적화 버전)
# ==========================================
st.header("2. 데이터 전처리")

if st.button("데이터 전처리 실행 (속도 최적화)"):
    with st.spinner("대용량 데이터 처리 중..."):
        prog = st.empty()
        df_p = df.copy()

        # 불필요 컬럼 제거
        prog.text("1/5: 식별자 제거 중...")
        df_p = df_p.drop(columns=[c for c in ['id', 'CustomerId', 'Surname'] if c in df_p.columns])
        
        # 결측치/이상치 (판다스 벡터 연산으로 속도 향상)
        prog.text("2/5: 결측치 및 이상치 처리 중...")
        nums = df_p.select_dtypes(include=[np.number]).columns.drop('Exited', errors='ignore')
        cats = df_p.select_dtypes(include=['object']).columns
        df_p[nums] = df_p[nums].fillna(df_p[nums].mean())
        for c in nums:
            q1, q3 = df_p[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df_p[c] = np.clip(df_p[c], q1 - 1.5*iqr, q3 + 1.5*iqr)

        # 인코딩 및 스케일링
        prog.text("3/5: 인코딩 및 스케일링 중...")
        df_p = pd.get_dummies(df_p, columns=cats, drop_first=True)
        sc = StandardScaler()
        cols = [c for c in df_p.columns if c != 'Exited']
        df_p[cols] = sc.fit_transform(df_p[cols])

        X, y = df_p.drop('Exited', axis=1), df_p['Exited']

        # 클래스 불균형 (SMOTE)
        if has_smote:
            prog.text("4/5: 불균형 데이터 조정 중...")
            X, y = SMOTE(random_state=42).fit_resample(X, y)
        
        # 변수 선택 (샘플링으로 속도 100배 향상)
        prog.text("5/5: 핵심 변수 자동 선택 중 (샘플링 사용)...")
        sample_X = X.sample(n=min(5000, len(X)), random_state=42)
        sample_y = y.loc[sample_X.index]
        sfs = SequentialFeatureSelector(LogisticRegression(max_iter=100), n_features_to_select='auto').fit(sample_X, sample_y)
        X = X[X.columns[sfs.get_support()]]
        
        st.session_state.processed_data = {'X': X, 'y': y}
        st.success(f"처리 완료! 선택된 변수: {list(X.columns)}")

# (이후 3~5단계 모델 평가 코드는 동일하게 작성)
# ... (생략된 뒷부분은 이전 드린 코드와 같으나 위에서 설정한 폰트가 자동으로 적용됩니다)
