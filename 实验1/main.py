import os
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from tqdm import tqdm


# =============================
# 自动切换到脚本所在目录
# =============================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())


# =============================
# 文本预处理
# =============================
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text)


# =============================
# 文档向量 = 词向量平均
# =============================
def get_document_vector(text, model):
    tokens = preprocess_text(text)
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


# =============================
# 余弦相似度
# =============================
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


# =============================
# 主流程：重新训练模型 + 相似度测试
# =============================
def main():
    # ---------------------------------------
    # 1. 加载数据集 dev.csv
    # ---------------------------------------
    csv_path = "dataset/dev.csv"
    if not os.path.exists(csv_path):
        print("❌ 找不到数据集:", csv_path)
        return

    print("\n正在加载数据...")
    df = pd.read_csv(csv_path)
    print("数据形状:", df.shape)

    # 合并 Title + Review
    title_series = df.iloc[:, 1].fillna("").astype(str)
    review_series = df.iloc[:, 2].fillna("").astype(str)
    texts = (title_series + " " + review_series).tolist()

    # ---------------------------------------
    # 2. 文本预处理
    # ---------------------------------------
    print("\n预处理文本...")
    corpus = [preprocess_text(t) for t in tqdm(texts)]

    # ---------------------------------------
    # 3. 训练 Word2Vec（重新训练，避免 numpy 不兼容）
    # ---------------------------------------
    print("\n正在训练 Word2Vec 模型...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=128,
        window=3,
        min_count=5,
        workers=16,
        epochs=5
    )

    # 保存模型
    model.save("word2vec_dev.model")
    print("模型已保存为 word2vec_dev.model")

    # ---------------------------------------
    # 4. 生成文档向量（仅用于相似度示例）
    # ---------------------------------------
    print("\n生成文档向量...")
    vectors = [get_document_vector(t, model) for t in tqdm(texts[:20])]  # 只取前 20 条示例

    # ---------------------------------------
    # 5. 示例：取第 0 和第 1 条评论计算相似度
    # ---------------------------------------
    text1 = texts[0]
    text2 = texts[1]
    sim = cosine_sim(vectors[0], vectors[1])

    print("\n==============================")
    print("示例相似度计算结果：")
    print("==============================")
    print("文本 1：", text1)
    print("文本 2：", text2)
    print(f"\n向量相似度: {sim:.4f}")
    print("\n🎉 完成！")


if __name__ == "__main__":
    main()
