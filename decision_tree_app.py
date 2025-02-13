import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

# タイトルとサブタイトルの表示
st.title("決定木分析ツール")
st.markdown("### データ分析と意思決定を支援")
st.markdown("**Decision Tree Analysis Tool** *| Supporting Data Analysis and Decision-Making*")

# 製作者の名前を右寄せで表示
st.markdown("""
<p style="text-align: right; font-size: 20px;">
<strong>作成者: 土居拓務（DOI, Takumu）</strong>
</p>
""", unsafe_allow_html=True)

# 利用上の注意や目的を表示
st.markdown("""
<div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px;">
本アプリを使用して分析する際は、以下の点にご留意ください。

1. **分析の目的**：本アプリは教育や研究用途での利用を目的としています。
2. **データの適切な前処理**：精度向上のため、欠損値や異常値を含むデータは適切に処理してください。
3. **結果の解釈**：本アプリの結果はデータに基づくものであり、必ずしも全てのケースにおいて最適な判断が得られるわけではありません。
</div>
""", unsafe_allow_html=True)

st.markdown("")  # 空の行を追加

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    # CSVファイルを読み込む
    data = pd.read_csv(uploaded_file)

    # 目的変数と説明変数に分ける
    y = data.iloc[:, 0]  # 最左列を目的変数
    X = data.iloc[:, 1:]  # 残りの列を説明変数

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 決定木モデルの作成と学習（max_depthを追加）
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # テストデータで予測し、精度を表示
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"モデルの精度: {accuracy:.2f}")

    # 特徴量重要度の表示
    st.write("特徴量の重要度:")
    feature_importances = pd.DataFrame({
        '特徴量': X.columns,
        '重要度': model.feature_importances_
    }).sort_values(by='重要度', ascending=False)
    st.write(feature_importances)

    # 樹形図のプロットサイズを動的に調整
    st.write("決定木の樹形図:")
    depth = model.get_depth()
    num_nodes = model.tree_.node_count
    fig_width = min(24, 10 + num_nodes // 8)
    fig_height = min(18, 8 + num_nodes // 12)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(model, feature_names=X.columns, filled=True, ax=ax, fontsize=10, proportion=False)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    st.pyplot(fig)

else:
    st.write("CSVファイルを上の枠内にドラッグ＆ドロップすれば分析が始まります。")

# 使用法の説明を枠で囲む
st.markdown("""
<div style="border: 2px solid black; padding: 10px; border-radius: 5px;">
<h3 style="font-weight: bold;">アプリの使用法</h3><br>  
このアプリは、CSVファイルを分析するためのツールです。使用方法は以下の通りです。
<br>

1. **CSVファイルの準備**:
   - 最左列には目的変数を記載してください。
   - 残りの列は説明変数として扱われます。
<br>

2. **列名の設定**:
   - 各変数の名称は最上列に記載されます。この列には、変数名を短めの半角英数字で記入することをお勧めします。
<br>

3. **分析の実行**:
   - CSVファイルをアップロードすると、自動的に目的変数と説明変数を分析します。
<br>

この手順に従ってCSVファイルを準備し、アプリを使用してください。
</div>
""", unsafe_allow_html=True)

st.markdown("")  # 空の行を追加

# 引用形式を強調して表示
st.markdown("""
**本アプリの利用に際しては、次のようにご記載いただけますと幸いです：**

<p style="text-align: right;">
DOI, Takumu (2024). Decision Tree Analysis Tool: Supporting Data Analysis and Decision-Making.
</p>
""", unsafe_allow_html=True)
