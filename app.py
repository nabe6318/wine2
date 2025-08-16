import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ワインデータ決定木（CSV/Streamlit）", layout="wide")
st.title("🍷 決定木 可視化アプリ（CSV対応 / Streamlit Cloud）")

st.markdown("""
- 左側で CSV をアップロードし、目的変数（ターゲット）と特徴量（説明変数）を選びます  
- `Graphviz` は不要（Streamlit Cloud で動作）  
- 決定木の**PNG画像**と**Graphviz DOT**をダウンロード可能
""")

# ---------- ユーティリティ ----------
def encode_features(df: pd.DataFrame, feature_cols):
    """特徴量の非数値列を LabelEncoder で数値化（学習用）。戻り値: X, encoders(dict)"""
    X = df[feature_cols].copy()
    encoders = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    return X, encoders

def encode_target(y_series: pd.Series):
    """目的変数をエンコード。カテゴリ/少数離散値は LabelEncoder、それ以外はそのまま（分類前提のチェック別）。"""
    class_names = None
    y_encoder = None

    # 目的変数のユニーク数
    nunique = y_series.nunique(dropna=True)

    # 非数値は LabelEncode
    if not pd.api.types.is_numeric_dtype(y_series):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y_series.astype(str))
        class_names = [str(c) for c in y_encoder.classes_]
        return y, y_encoder, class_names

    # 数値だが「離散っぽい」場合（クラス数が多すぎない）
    if nunique <= 20:
        # 0..K-1に並び替え
        classes_sorted = sorted(y_series.dropna().unique())
        mapping = {v: i for i, v in enumerate(classes_sorted)}
        y = y_series.map(mapping).values
        y_encoder = None
        class_names = [str(c) for c in classes_sorted]
        return y, y_encoder, class_names

    # 連続値っぽい（分類木には不向き）
    return None, None, None

# ---------- データ入力 ----------
with st.sidebar:
    st.header("📥 データ入力")
    uploaded = st.file_uploader("CSVをアップロード（ヘッダ行あり想定）", type=["csv"])
    use_sample = st.checkbox("サンプル（sklearn wine）を使う", value=(uploaded is None))

if use_sample:
    # sklearnのワインデータをそのままDataFrame化
    from sklearn.datasets import load_wine
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()  # features + target
    # target 名は数値。別途名前リストは wine.target_names
    st.caption("サンプル：sklearn wine データ")
else:
    if uploaded is None:
        st.info("CSVをアップロードするか、サンプルデータを使用してください。")
        st.stop()
    df = pd.read_csv(uploaded)

st.subheader("🔎 データプレビュー")
st.dataframe(df.head(), use_container_width=True)
st.write(f"行数: **{len(df)}**  列数: **{df.shape[1]}**")

# ---------- 列選択 ----------
all_cols = df.columns.tolist()
if len(all_cols) < 2:
    st.error("列が不足しています。CSVを確認してください。")
    st.stop()

default_target = "target" if "target" in all_cols else all_cols[-1]
target_col = st.selectbox("🎯 目的変数（分類クラス）", all_cols, index=all_cols.index(default_target))

feature_candidates = [c for c in all_cols if c != target_col]

# 「全選択/全解除」ボタン（セッションステートで保持）
if "feature_selection" not in st.session_state:
    st.session_state.feature_selection = feature_candidates.copy()

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🟩 全特徴量を選択"):
        st.session_state.feature_selection = feature_candidates.copy()
with col_b:
    if st.button("⬜ 全解除"):
        st.session_state.feature_selection = []

selected_features = st.multiselect(
    "🧮 使用する特徴量（複数選択）",
    options=feature_candidates,
    default=st.session_state.feature_selection,
    key="feature_selection",
)

if len(selected_features) == 0:
    st.warning("少なくとも1つの特徴量を選んでください。")
    st.stop()

# 欠損処理（簡易）：目的変数と選択特徴量に欠損がある行を除外
work_df = df[selected_features + [target_col]].dropna()
dropped = len(df) - len(work_df)
if dropped > 0:
    st.caption(f"⚠️ 欠損を含む {dropped} 行を削除しました（学習対象 {len(work_df)} 行）。")

# 目的変数の連続値チェック & エンコード
y_raw = work_df[target_col]
y, y_encoder, class_names = encode_target(y_raw)

if y is None:
    st.error("目的変数が連続値（クラス数が多すぎ）に見えます。分類木には**カテゴリ変数**を指定してください。")
    st.stop()

X, x_encoders = encode_features(work_df, selected_features)

# ---------- モデル設定 ----------
st.sidebar.header("🛠️ モデル設定")
criterion = st.sidebar.selectbox("不純度指標", ["gini", "entropy", "log_loss"], index=0)
max_depth = st.sidebar.slider("最大深さ（0=制限なし）", 0, 20, 3)
test_size = st.sidebar.slider("テストサイズ（割合）", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("random_state", 0, 9999, 0, step=1)
use_class_weight = st.sidebar.checkbox("クラス不均衡対策（class_weight='balanced')", value=False)

depth = None if max_depth == 0 else max_depth
class_weight = "balanced" if use_class_weight else None

# ---------- 学習 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=test_size, random_state=random_state, stratify=y
)

clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=depth,
    random_state=random_state,
    class_weight=class_weight,
)
clf.fit(X_train, y_train)

# ---------- 評価 ----------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 評価")
st.write(f"**Accuracy:** {acc:.4f}")

rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=False, zero_division=0)
st.code(rep)

# 混同行列（画像）
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
fig_cm, ax_cm = plt.subplots(figsize=(4 + 0.4*len(class_names), 4 + 0.4*len(class_names)))
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

# ---------- 決定木の可視化（matplotlib） ----------
st.subheader("🌳 決定木（matplotlib）")
fig, ax = plt.subplots(figsize=(min(24, 1.2*len(selected_features)+8), 8))
plot_tree(
    clf,
    feature_names=selected_features,
    class_names=class_names,
    filled=True,
    rounded=True,
    impurity=True,
    fontsize=10,
    ax=ax
)
st.pyplot(fig, use_container_width=True)

# PNG ダウンロード
buf = BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
png_bytes = buf.getvalue()
st.download_button(
    "📥 決定木PNGをダウンロード",
    data=png_bytes,
    file_name="decision_tree.png",
    mime="image/png"
)

# Graphviz DOT 文字列（画像化はしないが、後で使える）
dot_str = export_graphviz(
    clf,
    out_file=None,
    feature_names=selected_features,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)
st.download_button(
    "📥 Graphviz DOT をダウンロード",
    data=dot_str.encode("utf-8"),
    file_name="decision_tree.dot",
    mime="text/plain"
)

# ---------- 重要度 ----------
st.subheader("🏷️ 特徴量の重要度")
importances = clf.feature_importances_
imp_df = pd.DataFrame({"feature": selected_features, "importance": importances}).sort_values("importance", ascending=False)
st.dataframe(imp_df, use_container_width=True)

fig_imp, ax_imp = plt.subplots(figsize=(8, max(3, 0.4*len(selected_features))))
ax_imp.barh(imp_df["feature"], imp_df["importance"])
ax_imp.invert_yaxis()
ax_imp.set_xlabel("Importance")
st.pyplot(fig_imp, use_container_width=True)

# ---------- エンコードの注意 ----------
with st.expander("ℹ️ ラベルエンコードの対応（自動処理の説明）"):
    st.markdown("""
- 文字列の**特徴量**は LabelEncoder により 0..K-1 に自動変換しています  
- 目的変数（ターゲット）が文字列なら自動でクラスに変換します  
- 目的変数が数値でもクラス数が20以下なら**離散クラス**とみなして学習します  
- **多数の連続値**がある目的変数は分類木に不向きです（回帰木を使うのが望ましい）
""")
    if y_encoder is not None:
        map_df = pd.DataFrame({"class_index": list(range(len(y_encoder.classes_))),
                               "class_label": y_encoder.classes_})
        st.caption("目的変数のラベル→クラス番号 対応表")
        st.dataframe(map_df, use_container_width=True)
