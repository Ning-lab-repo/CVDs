import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv(r'X.csv')

# 删除缺失值
df = df.dropna(subset=['X'])
# 只保留 group==1 的数据
df['group'] = df['group'].astype(str)
df = df[df['group'] == '1']
# 只保留 cluster
cluster_order = ['X1', 'X2', 'X3']
df = df[df['cluster'].isin(cluster_order)]

# 设置 cluster 为有序分类
df['cluster'] = pd.Categorical(df['cluster'], categories=cluster_order, ordered=True)

# 筛选每个 ICD_prefixz 样本人群数 > 50 的
icd_counts = df['ICD'].value_counts()
valid_icds = icd_counts[icd_counts > 50].index
df = df[df['ICD'].isin(valid_icds)]

# 分组统计每个 ICD_prefixz 中各 cluster 的数量
counts = df.groupby(['ICD', 'cluster']).size().unstack(fill_value=0)

# 计算比例
proportions = counts.div(counts.sum(axis=1), axis=0)

# 绘图准备
icd_list = proportions.index.tolist()
x_pos = np.arange(len(icd_list))
bar_width = 0.8

# 自定义颜色
colors = ['#42B540', '#00468B','#ED0000']

# 开始绘图
fig, ax = plt.subplots(figsize=(12, 14), gridspec_kw={'bottom': 0.4})
bottoms = np.zeros(len(icd_list))

# 堆叠柱状图
for i, cluster in enumerate(cluster_order):
    values = proportions[cluster].values
    bars = ax.bar(x_pos, values, bottom=bottoms, color=colors[i % len(colors)],
                  label=cluster, width=bar_width, edgecolor='white')

    # 添加百分比标签
    for bar, val in zip(bars, values):
        if val > 0.01:  # 只标注 >1% 的值，避免太小太拥挤
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.2f}%",
                    ha='center', va='center', fontsize=18, color='black')

    bottoms += values
plt.rcParams['font.sans-serif'] = 'Arial'
# 设置 x 轴
ax.set_xticks(x_pos)
ax.set_xticklabels(icd_list, rotation=45, ha='right', fontsize=4)

# y轴标签
ax.set_ylabel('Proportion', fontsize=26)
ax.tick_params(axis='y', labelsize=26)
ax.set_xlabel('Disease', fontsize=26)
ax.tick_params(axis='x', labelsize=18)
# 图例
ax.legend(title='Group', fontsize=24, title_fontsize=24)

# 边框美化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.subplots_adjust(bottom=0.8)
# 保存为 PDF
fig.tight_layout()
fig.savefig(r'X.pdf', format='pdf', bbox_inches='tight')

plt.show()
