import pandas as pd
from fancyimpute import IterativeImputer

# 读取CSV文件
df = pd.read_csv(r'data.csv',engine = 'python',encoding='gb18030')

# 打印缺失值信息
print("缺失值情况：")
print(df.isnull().sum())
# 假设我们知道哪些列是ID和分类变量
id_vars = ['id']
binary_vars = []
categorical_vars = []

# 获取所有连续变量
continuous_vars = [col for col in df.columns if col not in id_vars + binary_vars + categorical_vars]

# 创建MICE对象
imputer = IterativeImputer(random_state=42)

# 对所有变量进行插补
df[continuous_vars + binary_vars + categorical_vars] = imputer.fit_transform(df[continuous_vars + binary_vars + categorical_vars])

# 确保二分类和多分类变量仍然是整数类型
df[binary_vars + categorical_vars] = df[binary_vars + categorical_vars].round().astype(int)
df = df.apply(lambda col: col.map(lambda x: max(x, 0) if isinstance(x, (int, float)) else x))

# 打印插补后的数据
print("插补后的数据：")
print(df)

# 将插补后的数据保存到新的CSV文件
df.to_csv(r'data1.csv', index=False)
