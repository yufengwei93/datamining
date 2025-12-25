import pandas as pd

# 原始训练集路径
input_path = "dataset/train.csv"
# 输出的新训练集路径
output_path = "dataset/train_filtered.csv"

# 读取数据
df = pd.read_csv(input_path, header=None)

# 标签列是第一列 df[0]
#   好评 = 2
#   差评 = 1

good = df[df[0] == 2].head(1000)  # 选好评 1000 条
bad = df[df[0] == 1].head(1000)   # 选差评 1000 条

# 合并
filtered = pd.concat([good, bad], axis=0)

# 打乱顺序（非常重要）
filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存新训练集
filtered.to_csv(output_path, header=False, index=False)

print("筛选完成！")
print(f"好评数量: {len(good)}")
print(f"差评数量: {len(bad)}")
print(f"合并后总数: {len(filtered)}")
print("新数据集已保存为:", output_path)
