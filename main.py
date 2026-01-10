import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

import matplotlib.font_manager as fm
import platform

# 폰트 설정 함수
def set_korean_font():
    system_name = platform.system()
    
    if system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Windows': # Windows
        plt.rc('font', family='Malgun Gothic')
    else: # Linux (Streamlit Cloud 포함)
        # 1단계에서 설치한 나눔 고딕을 설정
        try:
            # 폰트 경로를 직접 지정하거나 폰트 이름으로 설정
            plt.rc('font', family='NanumGothic')
        except:
            # 만약 폰트 이름으로 안될 경우를 대비
            st.warning("나눔 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 앱 시작 부분에서 실행
set_korean_font()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

# SMOTE 라이브러리 확인 (불균형 데이터 처리)
try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False
    st.warning("imbalanced-learn 라이브러리가 설치되지 않았습니다. SMOTE 기능은 건너뜁니다. (설치 명령어: pip install imbalanced-learn)")

# 페이지 설정
st.set_page_config(page_title="고객 이탈 예측 앱", layout="wide")

# 제목
st.title("📊 고객 이탈 관리 데이터 분석 앱")

# 1. 데이터 불러오기
@st.cache_data
def load_data():
    # 파일이 같은 디렉토리에 있다고 가정
    df = pd.read_csv("Churn_management.csv")
    return df

try:
    df = load_data()
    st.success("데이터를 성공적으로 불러왔습니다!")
except FileNotFoundError:
    st.error("'Churn_management.csv' 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜 주세요.")
    st.stop()

# 세션 상태 초기화 (전처리된 데이터 저장용)
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None

# ==========================================
# 1. 데이터 분석 및 시각화
# ==========================================
st.header("1. 데이터 분석 및 시각화")

# 원본 데이터 보기
if st.checkbox("원본 데이터 보기"):
    st.dataframe(df.head())

# 종속 변수 분포 확인
st.subheader("종속 변수 분포 (이탈 여부: Exited)")
fig_target, ax_target = plt.subplots(figsize=(6, 4))
sns.countplot(x='Exited', data=df, ax=ax_target, palette='viridis')
ax_target.set_title("이탈 여부 분포 (0: 유지, 1: 이탈)")
st.pyplot(fig_target)

# 인터랙티브 시각화
st.subheader("변수별 시각화")
col1, col2, col3 = st.columns(3)

with col1:
    plot_type = st.selectbox("그래프 유형 선택", ["히스토그램", "박스 플롯", "산점도", "막대 차트", "선 차트"])
with col2:
    x_var = st.selectbox("X축 변수 선택", df.columns)
with col3:
    y_var = st.selectbox("Y축 변수 선택 (히스토그램/박스플롯은 선택 사항)", [None] + list(df.columns))

fig_custom, ax_custom = plt.subplots(figsize=(8, 5))

try:
    if plot_type == "히스토그램":
        sns.histplot(data=df, x=x_var, kde=True, ax=ax_custom)
    elif plot_type == "박스 플롯":
        sns.boxplot(data=df, x=x_var, y=y_var, ax=ax_custom)
    elif plot_type == "산점도":
        if y_var:
            sns.scatterplot(data=df, x=x_var, y=y_var, hue='Exited', ax=ax_custom)
        else:
            st.warning("산점도를 그리려면 Y축 변수를 선택해야 합니다.")
    elif plot_type == "막대 차트":
        if y_var:
            sns.barplot(data=df, x=x_var, y=y_var, ax=ax_custom)
        else:
            sns.countplot(data=df, x=x_var, ax=ax_custom)
    elif plot_type == "선 차트":
        if y_var:
            sns.lineplot(data=df, x=x_var, y=y_var, ax=ax_custom)
        else:
            st.warning("선 차트를 그리려면 Y축 변수를 선택해야 합니다.")
    
    st.pyplot(fig_custom)
except Exception as e:
    st.error(f"그래프를 그리는 중 오류가 발생했습니다: {e}")


# ==========================================
# 2. 데이터 전처리
# ==========================================
st.header("2. 데이터 전처리")

if st.button("데이터 전처리 실행"):
    with st.spinner("데이터 전처리 및 특성 선택 중..."):
        df_processed = df.copy()

        # 1. 불필요한 식별자 컬럼 제거
        drop_cols = ['id', 'CustomerId', 'Surname']
        df_processed = df_processed.drop(columns=[c for c in drop_cols if c in df_processed.columns])
        
        # 2. 결측치 처리 (수치형: 평균, 범주형: 최빈값)
        num_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        
        # 타겟 변수 분리 (Exited가 수치형 컬럼에 포함되지 않도록)
        if 'Exited' in num_cols:
            num_cols = num_cols.drop('Exited')

        imputer_num = SimpleImputer(strategy='mean')
        df_processed[num_cols] = imputer_num.fit_transform(df_processed[num_cols])
        
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_processed[cat_cols] = imputer_cat.fit_transform(df_processed[cat_cols])

        # 3. 이상치 처리 (IQR 방식 - Clipping)
        for col in num_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

        # 4. 원-핫 인코딩 (One-Hot Encoding)
        df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)

        # 5. 특성 스케일링 (StandardScaler)
        scaler = StandardScaler()
        cols_to_scale = [c for c in df_processed.columns if c != 'Exited']
        df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])

        # X, y 분리
        X = df_processed.drop('Exited', axis=1)
        y = df_processed['Exited']

        # 6. 클래스 불균형 처리 (SMOTE)
        if has_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        # 7. 단계적 선택법 (Stepwise Selection)
        # 속도를 위해 로지스틱 회귀 기본 모델 사용
        sfs = SequentialFeatureSelector(LogisticRegression(max_iter=1000), 
                                        n_features_to_select='auto', 
                                        direction='forward',
                                        tol=None)
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()]
        X = X[selected_features]
        
        # 세션 상태에 저장
        st.session_state.processed_data = {'X': X, 'y': y}
        st.success("전처리 완료! 모델링 준비가 되었습니다.")
        st.write(f"**선택된 특성:** {list(selected_features)}")
        st.write(f"**전처리 후 데이터 크기:** {X.shape}")


# ==========================================
# 3. 데이터 분할 및 모델 설정
# ==========================================
st.header("3. 데이터 분할 및 모델 설정")

if st.session_state.processed_data is not None:
    X = st.session_state.processed_data['X']
    y = st.session_state.processed_data['y']

    # 분할 비율 선택
    split_ratio = st.radio("학습/테스트 데이터 분할 비율 선택", ["7:3", "8:2"])
    test_size = 0.3 if split_ratio == "7:3" else 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    col_dt, col_log = st.columns(2)

    with col_dt:
        st.subheader("의사결정나무(Decision Tree) 옵션")
        dt_max_depth = st.slider("최대 깊이 (Max Depth)", 1, 20, 5)
        dt_criterion = st.selectbox("분할 기준 (Criterion)", ["gini", "entropy"])

    with col_log:
        st.subheader("로지스틱 회귀(Logistic Regression) 옵션")
        lr_C = st.slider("C 값 (규제 강도의 역수)", 0.01, 10.0, 1.0)
        lr_max_iter = st.number_input("최대 반복 횟수 (Max Iterations)", value=500)

    # 학습 및 평가 버튼
    if st.button("모델 학습 및 평가"):
        
        # 평가지표 출력 헬퍼 함수
        def show_metrics(y_true, y_pred, y_prob, title):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            st.markdown(f"### **{title} 평가 지표**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("정확도 (Accuracy)", f"{acc:.2f}")
            m2.metric("정밀도 (Precision)", f"{prec:.2f}")
            m3.metric("재현율 (Recall)", f"{rec:.2f}")
            m4.metric("F1 점수", f"{f1:.2f}")

            # 그래프 출력
            c1, c2 = st.columns(2)
            
            # Confusion Matrix
            with c1:
                st.write("**혼동 행렬 (Confusion Matrix)**")
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("예측값")
                ax_cm.set_ylabel("실제값")
                st.pyplot(fig_cm)
            
            # ROC Curve
            with c2:
                st.write("**ROC 곡선**")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate (위양성률)')
                ax_roc.set_ylabel('True Positive Rate (진양성률)')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

        # ==========================================
        # 4. 모델 평가 (의사결정나무)
        # ==========================================
        st.header("4. 모델 평가 - 의사결정나무")
        dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, criterion=dt_criterion, random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
        
        show_metrics(y_test, y_pred_dt, y_prob_dt, "의사결정나무")

        st.markdown("---") # 구분선

        # ==========================================
        # 5. 모델 평가 (로짓 모델)
        # ==========================================
        st.header("5. 모델 평가 - 로지스틱 회귀")
        lr_model = LogisticRegression(C=lr_C, max_iter=int(lr_max_iter), random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

        show_metrics(y_test, y_pred_lr, y_prob_lr, "로지스틱 회귀")

else:
    st.info("👆 먼저 2번 섹션의 '데이터 전처리 실행' 버튼을 눌러주세요.")
