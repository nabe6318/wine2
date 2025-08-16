import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="ワインデータ決定木（Streamlit）", layout="wide")
st.title("🍷 ワインデータの決定木学習（scikit-learn）")

# -------------------------------------------------
# データ読み込み（CSVは使わない）
# -------------------------------------------------
wine = load_wine()
X = wine.data              # ← ご指定通り
y = wine.target            # ← ご指定通り
feature_names = wine.feature_names
class_names = wine.target_names

# 表示用 DataFrame（先頭5行）
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
st.subheader("データセット先頭5行")
st.dataframe(df.head(5), use_container_width=True)

# -------------------------------------------------
# 説明変数の選択（初期は全選択）
# -------------------------------------------------
st.subheader("説明変数の選択")
col_btn1, col_btn2 = st.columns(2)
if "selected_features" not in st.session_state:
    st.session_state.selected_features = feature_names.copy()

with col_btn1:
    if st.button("🟩 全選択"):
        st.session_state.selected_features = feature_names.copy()
with col_btn2:
    if st.button("⬜ 全解除"):
        st.session_state.selected_features = []

selected_features = st.multiselect(
    "使用する説明変数",
    options=feature_names,
    default=st.session_state.selected_features,
    key="selected_features"
)

if len(selected_features) == 0:
    st.warning("少なくとも1つの説明変数を選択してください。")
    st.stop()

# -------------------------------------------------
# モデル設定（ジニ係数/深さ）
# -------------------------------------------------
st.subheader("モデル設定")
criterion = st.radio(
    "不純度指標（ジニ係数/エントロピー）",
    options=["gini", "entropy"],
    index=0,
    help="通常は gini（ジニ係数）でOKです。"
)
max_depth = st.number_input("決定木の最大深さ（0=制限なし）", min_value=0, max_value=50, value=3, step=1)
random_state = st.number_input("random_state（再現性）", min_value=0, max_value=9999, value=0, step=1)

depth = None if max_depth == 0 else int(max_depth)

# -------------------------------------------------
# 学習・評価
# -------------------------------------------------
st.subheader("学習・評価")
# 使う説明変数だけ取り出し
X_use = pd.DataFrame(X, columns=feature_names)[selected_features].values

X_train, X_test, y_train, y_test = train_test_split(
    X_use, y, test_size=0.2, random_state=int(random_state), stratify=y
)
clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=int(random_state))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.4f}")

rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
st.code(rep, language="text")

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

# -------------------------------------------------
# 決定木の可視化（ダウンロード機能なし）
# -------------------------------------------------
st.subheader("決定木（可視化）")
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

# -------------------------------------------------
# 特徴量重要度
# -------------------------------------------------
st.subheader("特徴量の重要度")
importances = clf.feature_importances_
imp_df = pd.DataFrame({"feature": selected_features, "importance": importances}).sort_values("importance", ascending=False)
st.dataframe(imp_df, use_container_width=True)


