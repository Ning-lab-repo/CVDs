import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'data.csv',engine='python',encoding='gb18030')
def mark_column1(row):
    # 获取特定列中的值
    value = row['教育程度']

    # 根据不同条件返回不同的值
    if value == 1 or value == 2:
        return 1
    elif value == 3:
        return 2
    elif value == 4 or value == 5:
        return 3
    else:
        return value  # 如果没有匹配的条件，默认返回0或者根据需求返回其他值

# 打印处理后的数据，包括新列
df['教育程度分类分类'] = df.apply(mark_column1, axis=1)
print(f'Total rows in df: {len(df)}')
print(df.head())
df.to_csv(r'data1.csv', index=True,encoding='utf_8_sig')




import pandas as pd

# 读取CSV文件
data = pd.read_csv(r'data.csv',engine = 'python')

# 定义函数判断是否满足高血压条件
def is_htn(row):
    if (row['SBP'] >= 140 or
        row['DBP'] >= 90 or
        row['高血压现病史'] == 1 or
        row['用药史'] == 1):
        return 1
    else:
        return 2

# 应用函数创建新列
data['高血压'] = data.apply(is_htn, axis=1)
data.to_csv(r'data1.csv', index=False)
