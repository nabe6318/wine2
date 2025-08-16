import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(page_title="🍷 決定木（CSV読み込み／ステップ実行）", layout="wide")
st.title("🍷 決定木学習（CSV読み込み / ステップ実行）")

st.markdown("""
**使い方（順番に実行）**  
1) サイドバーでCSVをアップロード（ヘッダ行あり）。  
2) 目的変数（ターゲット）は自動検出します（UIには表示しません）。  
3) 使用する説明変数を選択（初期値は全選択）。  
4) 不純度指標（ジニ係数/エントロピー）と決定木の最大深さを設定。  
5) 学習を実行して結果を確認。  
""")

# -----------------------------
# セッション状態（ステップ管理）
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

def go_next():
    st.session_state.step += 1

def go_prev():
    st.session_state.step = max(1, st.session_state.step - 1)

# -----------------------------
# 目的変数の自動検出
# -----------------------------
TARGET_CANDIDATES = ["target", "class", "label", "y", "species"]

def detect_target_column(df: pd.DataFrame) -> str | None:
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    # 見つからなければ最終列をターゲットとみなす（明示的にCSV末尾に置いてください）
    if df.shape[1] >= 2:
        return df.columns[-1]
    return None

def encode_target(y_series: pd.Series):
    """目的変数を分類用にエンコード。連続値（クラス数>20など）はエラーを返す。"""
    nunique = y_series.nunique(dropna=True)
    class_names = None
    y_encoder = None

    # 文字列 or カテゴリ → LabelEncode
    if not pd.api.types.is_numeric_dtype(y_series):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y_series.astype(str))
        class_names = [str(c) for c in y_encoder.classes_]
        return y, class_names, y_encoder

    # 数値：クラス数が少なければ離散ラベルとみなす
    if nunique <= 20:
        classes_sorted = sorted(y_series.dropna().unique())
        mapping = {v: i for i, v in enumerate(classes_sorted)}
        y = y_series.map(mapping).values
        class_names = [str(c) for c in classes_sorted]
        return y, class_names, None

    # 連続値っぽい → 分類木には不適
    return None, None, None

def encode_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    encoders = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    return X, encoders


# -----------------------------
# ステップ1: CSVアップロード & ターゲット自動検出
# -----------------------------
st.sidebar.header("📥 ステップ1：CSVアップロード")
uploaded = st.sidebar.file_uploader("CSVファイル（ヘッダ行あり）を選択", type=["csv"])

if st.session_state.step == 1:
    st.subheader("ステップ1：データの読み込み")
    if uploaded is None:
        st.info("CSV をアップロードしてください。")
    else:
        df = pd.read_csv(uploaded)
        st.write(f"行数: **{len(df)}**, 列数: **{df.shape[1]}**")
        st.dataframe(df.head(), use_container_width=True)

        target_col = detect_target_column(df)
        if target_col is None:
            st.error("目的変数（ターゲット）を自動検出できませんでした。"
                     f" 列名に {TARGET_CANDIDATES} のいずれかを使うか、ターゲット列をファイルの最終列に置いてください。")
        else:
            st.success(f"目的変数（ターゲット）: **{target_col}**（自動検出）")
            st.caption("※ 目的変数は特徴量選択から自動的に除外されます。UIには表示しません。")
            if st.button("次へ ▶"):
                st.session_state.df = df
                st.session_state.target_col = target_col
                st.session_state.step = 2

# -----------------------------
# ステップ2: 特徴量選択
# -----------------------------
if st.session_state.step >= 2 and "df" in st.session_state:
    df = st.session_state.df
    target_col = st.session_state.target_col
    st.subheader("ステップ2：使用する説明変数の選択")

    feature_candidates = [c for c in df.columns if c != target_col]

    # 初期値は全選択
    default_features = feature_candidates.copy()
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = default_features

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟩 全選択"):
            st.session_state.selected_features = feature_candidates.copy()
    with col2:
        if st.button("⬜ 全解除"):
            st.session_state.selected_features = []

    selected_features = st.multiselect(
        "説明変数（複数選択）", options=feature_candidates,
        default=st.session_state.selected_features, key="selected_features"
    )

    if len(selected_features) == 0:
        st.warning("少なくとも1つの説明変数を選んでください。")
    colp = st.columns(2)
    with colp[0]:
        if st.button("◀ 戻る"):
            go_prev()
    with colp[1]:
        if st.button("次へ ▶", disabled=(len(selected_features) == 0)):
            st.session_state.step = 3

# -----------------------------
# ステップ3: モデル設定（ジニ係数・深さ）
# -----------------------------
if st.session_state.step >= 3 and "df" in st.session_state:
    st.subheader("ステップ3：モデル設定")
    st.caption("不純度指標はデフォルトで **gini（ジニ係数）** を使用します。必要に応じて変更してください。")

    with st.form("model_settings"):
        criterion = st.radio("不純度指標", options=["gini", "entropy"], index=0,
                             help="gini=ジニ係数, entropy=情報利得")
        max_depth = st.number_input("決定木の最大深さ（0=制限なし）", min_value=0, max_value=50, value=3, step=1)
        random_state = st.number_input("random_state（再現性）", min_value=0, max_value=9999, value=0, step=1)
        submitted = st.form_submit_button("設定を確定して次へ ▶")

    if submitted:
        st.session_state.criterion = criterion
        st.session_state.max_depth = max_depth
        st.session_state.random_state = random_state
        st.session_state.step = 4

    if st.button("◀ 戻る（特徴量選択へ）"):
        go_prev()

# -----------------------------
# ステップ4: 学習・評価・可視化
# -----------------------------
if st.session_state.step >= 4 and "df" in st.session_state:
    st.subheader("ステップ4：学習・評価・可視化")

    df = st.session_state.df
    target_col = st.session_state.target_col
    selected_features = st.session_state.selected_features
    criterion = st.session_state.criterion
    depth = None if st.session_state.max_depth == 0 else int(st.session_state.max_depth)
    random_state = int(st.session_state.random_state)

    # 欠損除去（目的変数＋説明変数）
    work_df = df[selected_features + [target_col]].dropna()
    dropped = len(df) - len(work_df)
    if dropped > 0:
        st.caption(f"⚠️ 欠損値を含む {dropped} 行を除外しました。学習対象: {len(work_df)} 行")

    # 目的変数エンコード
    y_raw = work_df[target_col]
    y, class_names, y_encoder = encode_target(y_raw)
    if y is None:
        st.error("目的変数が連続値（クラス数が多すぎ）と判断されました。分類木には不適です。"
                 " ターゲットはカテゴリ（または少数の離散値）にしてください。")
        st.stop()

    # 説明変数エンコード
    X, _ = encode_features(work_df, selected_features)

    # 学習
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = DecisionTreeClassifier(
        criterion=criterion, max_depth=depth, random_state=random_state
    )
    clf.fit(X_train, y_train)

    # 評価
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**Accuracy:** `{acc:.4f}`")

    rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    st.code(rep, language="text")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fig_cm, ax_cm = plt.subplots(figsize=(5 + 0.5*len(class_names), 5 + 0.5*len(class_names)))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, str(v), ha='center', va='center')
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    st.pyplot(fig_cm, use_container_width=True)

    # 決定木の可視化（ダウンロードは提供しない）
    st.markdown("### 決定木（可視化）")
    fig_tree, ax_tree = plt.subplots(figsize=(min(24, 1.5*len(selected_features)+6), 8))
    plot_tree(
        clf,
        feature_names=selected_features,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=10,
        ax=ax_tree
    )
    st.pyplot(fig_tree, use_container_width=True)

    # 特徴量重要度
    st.markdown("### 特徴量の重要度")
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"feature": selected_features, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(imp_df, use_container_width=True)

    if st.button("◀ 設定に戻る"):
        st.session_state.step = 3

