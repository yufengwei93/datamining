import os
import numpy as np
from gensim.models import Word2Vec

# 自动切换到脚本目录（避免路径问题）
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())


# 余弦相似度
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def main():
    # 加载你的模型
    model_path = "word2vec_dev.model"
    print(f"正在加载模型 {model_path} ...")
    model = Word2Vec.load(model_path)
    print("模型加载成功！\n")

    while True:
        print("请输入两个要比较的词（输入 q 退出）：")
        w1 = input("词 1：").strip()
        if w1 == "q":
            break
        w2 = input("词 2：").strip()
        if w2 == "q":
            break

        # 是否存在词典中
        if w1 not in model.wv:
            print(f"❌ '{w1}' 不在模型词典中！请换一个词。\n")
            continue

        if w2 not in model.wv:
            print(f"❌ '{w2}' 不在模型词典中！请换一个词。\n")
            continue

        # 计算相似度
        v1 = model.wv[w1]
        v2 = model.wv[w2]
        sim = cosine_sim(v1, v2)

        print(f"\n'{w1}' 和 '{w2}' 的相似度 = {sim:.4f}\n")
        print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
