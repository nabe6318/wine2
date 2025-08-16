import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="ワイン決定木 | Wine Decision Tree (Step-by-Step)", layout="wide")
st.title("🍷 ワイン決定木（ステップ実行） / Wine Decision Tree (Step-by-Step)")

st.markdown("""
**使い方 / How to use**
1) **Step 1:** データの確認（先頭5行＆全テーブル） / Preview the dataset (first 5 rows & full table)  
2) **Step 2:** 説明変数を選択 / Select explanatory variables (features)  
3) **Step 3:** モデル設定（ジニ係数・深さなど） / Set model options (criterion & depth)  
4) **Step 4:** 学習・評価・可視化 / Train, evaluate, and visualize the tree
""")

# -------------------------------------------------
# Session state (step control)
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

def go(step: int):
    st.session_state.step = step

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step = max(1, st.session_state.step - 1)

# -------------------------------------------------
# Load data (Wine only)
# -------------------------------------------------
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
class_names = wine.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# -------------------------------------------------
# Progress indicator
# -------------------------------------------------
st.progress((st.session_state.step - 1) / 3)

# =========================
# STEP 1: Data preview
# =========================
if st.session_state.step == 1:
    st.header("Step 1: データの確認 / Data Preview")

    with st.expander("説明 / Explanation", expanded=True):
        st.markdown("""
- **日本語**: このステップでは、学習に使用するワインデータセットの**先頭5行**と、**全データ（テーブル）**を表示します。  
  目的変数 `target` はワインの**クラス（3種類）**を表します。
- **English**: In this step, we show the **first five rows** and the **entire dataset (table)**.  
  The `target` column indicates the wine **class (3 types)**.
""")

    st.caption(f"行数 / Rows: **{len(df)}**　列数 / Columns: **{df.shape[1]}**")

    # 先頭5行（固定プレビュー）
    st.subheader("先頭5行 / First 5 rows")
    st.table(df.head(5))

    # 全データ（スクロール可能なテーブル）
    st.subheader("全データ（スクロール可）/ Full dataset (scrollable)")
    st.dataframe(df, use_container_width=True, height=500)

    c1, c2 = st.columns([1,1])
    with c1:
        st.button("▶ 次へ / Next", on_click=next_step, key="s1_next")
    with c2:
        st.button("🔄 リセット / Reset", on_click=lambda: go(1), key="s1_reset")

# =========================
# STEP 2: Feature selection
# =========================
if st.session_state.step == 2:
    st.header("Step 2: 説明変数の選択 / Select Features")

    with st.expander("説明 / Explanation", expanded=True):
        st.markdown("""
- **日本語**: 使用する説明変数（特徴量）を選びます。初期設定は「全て選択」です。  
  少なくとも1つは選んでください。
- **English**: Choose the explanatory variables (features) to use.  
  All features are selected by default. Please select **at least one**.
""")

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = feature_names.copy()

    colA, colB = st.columns(2)
    with colA:
        if st.button("🟩 全選択 / Select All", key="s2_select_all"):
            st.session_state.selected_features = feature_names.copy()
    with colB:
        if st.button("⬜ 全解除 / Clear All", key="s2_clear_all"):
            st.session_state.selected_features = []

    selected_features = st.multiselect(
        "使用する説明変数 / Features to use",
        options=feature_names,
        default=st.session_state.selected_features,
        key="s2_multiselect",
    )
    # Keep session in sync
    st.session_state.selected_features = selected_features

    nav1, nav2 = st.columns([1,1])
    with nav1:
        st.button("◀ 戻る / Back", on_click=prev_step, key="s2_back")
    with nav2:
        st.button("▶ 次へ / Next", on_click=next_step, disabled=(len(selected_features) == 0), key="s2_next")

# =========================
# STEP 3: Model settings
# =========================
if st.session_state.step == 3:
    st.header("Step 3: モデル設定 / Model Settings")

    with st.expander("説明 / Explanation", expanded=True):
        st.markdown("""
- **日本語**: 不純度指標（ジニ係数 *gini* または エントロピー *entropy*）と、決定木の最大深さを設定します。  
  深さが大きいほど複雑な木になり、過学習のリスクが高まります。
- **English**: Set the impurity criterion (*gini* or *entropy*) and the maximum depth of the tree.  
  Larger depth increases model complexity and the risk of overfitting.
""")

    with st.form("model_settings"):
        criterion = st.radio(
            "不純度指標 / Criterion",
            options=["gini", "entropy"],
            index=0,
            help="gini=ジニ係数, entropy=情報利得 / gini=Gini impurity, entropy=Information gain",
            key="s3_criterion",
        )
        max_depth = st.number_input(
            "決定木の最大深さ（0=制限なし）/ Max depth (0 = unlimited)",
            min_value=0, max_value=50, value=3, step=1, key="s3_max_depth"
        )
        random_state = st.number_input(
            "random_state（再現性）/ random_state",
            min_value=0, max_value=9999, value=0, step=1, key="s3_random_state"
        )
        submitted = st.form_submit_button("設定を確定して次へ / Apply & Next ▶", use_container_width=True)

    if submitted:
        st.session_state.criterion = criterion
        st.session_state.max_depth = max_depth
        st.session_state.random_state = random_state
        next_step()

    st.button("◀ 戻る / Back", on_click=prev_step, key="s3_back")

# =========================
# STEP 4: Train & evaluate
# =========================
if st.session_state.step == 4:
    st.header("Step 4: 学習・評価・可視化 / Train, Evaluate & Visualize")

    with st.expander("説明 / Explanation", expanded=True):
        st.markdown("""
- **日本語**: データを 80/20 に分割し、決定木で学習します。  
  精度、分類レポート、混同行列、決定木の図、特徴量重要度を表示します。
- **English**: The data are split into 80/20 for training/testing, and a decision tree is trained.  
  We show accuracy, classification report, confusion matrix, the tree plot, and feature importances.
""")

    # Gather settings & data
    selected_features = st.session_state.get("selected_features", feature_names)
    if len(selected_features) == 0:
        st.error("説明変数が選択されていません。Step 2 に戻って選択してください。 / No features selected. Please go back to Step 2.")
        st.stop()

    criterion = st.session_state.get("criterion", "gini")
    depth = None if st.session_state.get("max_depth", 3) == 0 else int(st.session_state.get("max_depth", 3))
    random_state = int(st.session_state.get("random_state", 0))

    # Build X with selected features
    X_use = pd.DataFrame(X, columns=feature_names)[selected_features].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("結果 / Results")
    st.write(f"**Accuracy:** {acc:.4f}")

    st.caption("分類レポート / Classification report")
    rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    st.code(rep, language="text")

    st.caption("混同行列 / Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, str(v), ha="center", va="center")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    st.pyplot(fig_cm, use_container_width=True)

    st.caption("決定木の可視化 / Decision Tree plot")
    fig_tree, ax_tree = plt.subplots(figsize=(min(24, 1.5 * len(selected_features) + 6), 8))
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

    st.caption("特徴量重要度 / Feature Importances")
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"feature": selected_features, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(imp_df, use_container_width=True)

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.button("◀ 戻る（設定）/ Back (Settings)", on_click=prev_step, key="s4_back")
    with nav2:
        st.button("⏮ 最初に戻る / Back to Step 1", on_click=lambda: go(1), key="s4_to_s1")
    with nav3:
        st.button("🔄 再実行 / Rerun", on_click=lambda: go(4), key="s4_rerun")



