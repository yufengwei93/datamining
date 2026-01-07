import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# =============================
# 自动切换到脚本所在目录
# =============================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())


def visualize_tsne(model_path="word2vec_dev.model", num_words=500):
    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    model = Word2Vec.load(model_path)
    print("模型加载成功！")

    # 2. 获取前 num_words 个高频词
    vocab = list(model.wv.index_to_key)[:num_words]
    print(f"选择 {num_words} 个词用于可视化...")

    # 3. 获取这些词的向量
    word_vectors = np.array([model.wv[word] for word in vocab])

    # 4. 使用 t-SNE 降维到二维
    print("t-SNE 降维中，可能需要几秒...")
    tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", init='pca')
    vectors_2d = tsne.fit_transform(word_vectors)

    # 5. 可视化
    plt.figure(figsize=(12, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=30, alpha=0.7)

    # 添加标签
    for i, word in enumerate(vocab):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9, alpha=0.8)

    plt.title("t-SNE Visualization of Word2Vec Embeddings")
    plt.show()


if __name__ == "__main__":
    visualize_tsne()
