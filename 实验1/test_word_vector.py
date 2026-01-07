from gensim.models import Word2Vec

# =============================
# 加载模型
# =============================
model_path = "word2vec_dev.model"
model = Word2Vec.load(model_path)
print("模型加载成功:", model_path)

# =============================
# 查询词向量
# =============================
def get_word_vector(word):
    word = word.lower().strip()
    if word in model.wv:
        return model.wv[word]
    else:
        print(f"❌ 单词 '{word}' 不在词表中")
        return None

# =============================
# 主流程：手动输入词
# =============================
while True:
    word = input("\n请输入一个单词（exit 退出）：")
    if word.lower() == "exit":
        print("退出程序。")
        break

    vec = get_word_vector(word)
    if vec is not None:
        print(f"\n【{word}】的词向量（前10维示例）：")
        print(vec[:128])  # 只打印前10维防止太长
        print(f"\n向量维度：{len(vec)}")
