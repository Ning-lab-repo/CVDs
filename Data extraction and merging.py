#提取多列并进行竖向合并
import pandas as pd
import os
import glob
# 文件路径列表
file_paths = glob.glob(r'file\*.csv')  # 将实际文件路径替换为file1.csv, file2.csv, file3.csv

# 要提取的列
columns_to_extract = []  # 将实际列名替换为column1, column2

# 初始化一个空的DataFrame用于存储合并后的数据
merged_df = pd.DataFrame()

# 遍历每个文件
for file_path in file_paths:
    # 读取CSV文件
    df = pd.read_csv(file_path, engine='python',encoding='gb18030',usecols=columns_to_extract)
    # 将当前文件的DataFrame竖向合并到总的DataFrame中
    merged_df = pd.concat([merged_df, df], axis=0)

# 重置索引
merged_df.reset_index(drop=True, inplace=True)

# 输出合并后的DataFrame
print(merged_df)

# 将合并后的DataFrame保存为新的CSV文件
merged_df.to_csv(r'data.csv', index=False)

#列名不一致时进行合并
import pandas as pd
import glob

# Step 1: 读取所有的 CSV 文件并存储为 DataFrame 列表
csv_files = glob.glob(r'flie\*.csv')
dfs = [pd.read_csv(file,engine='python',encoding='gb18030') for file in csv_files]

# Step 2: 统一行索引（假设索引列为 'index_column'）
for i in range(len(dfs)):
    # 检查当前 DataFrame 是否有指定的索引列
    if 'index_column' in dfs[i].columns:
        dfs[i] = dfs[i].set_index('index_column')
    else:
        # 如果没有指定的索引列，可以选择添加一个默认的索引列或者根据具体情况处理
        dfs[i] = dfs[i].reset_index(drop=True)  # 重置索引，使用默认整数索引

# Step 3: 合并 DataFrame
merged_df = pd.concat(dfs, axis=0)

# 可选：保存合并后的 DataFrame 为 CSV 文件
merged_df.to_csv(r'data.csv')


import pandas as pd
df1 = pd.read_csv(r"data1.csv",engine = 'python',encoding='gb18030')
df2 = pd.read_csv(r"data2.csv",engine = 'python')
data3=pd.merge(df1,df2,on=["id"],how='left')
data3.to_csv(r'data3.csv',encoding='utf_8_sig')
