import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from statsmodels.stats.multitest import multipletests
import itertools

# 读取CSV文件
data = pd.read_csv('data.csv')

# 去除 HTN 列的空白值并清理可能的空格
data_cleaned = data.dropna(subset=['Feature'])
# 正确使用 .loc 进行赋值，确保操作的是原始 DataFrame
data_cleaned.loc[:, 'Feature'] = data_cleaned['Feature'].astype(str).str.strip()

# 假设csv文件中有'Group'和一个二分类指标列'Binary_Indicator'
group_column = 'Cluster'
binary_indicator_column = 'Feature'  # 替换为实际二分类指标列名

# 计算每组中二分类指标的占比
binary_counts = data_cleaned.groupby([group_column, binary_indicator_column]).size().unstack(fill_value=0)
binary_ratios = binary_counts.div(binary_counts.sum(axis=1), axis=0) * 100  # 转换为百分比

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'

# 绘制二分类指标占比的柱状图
fig, ax = plt.subplots(figsize=(6, 8))
colors = ['#1f77b4', '#ff7f0e']
binary_ratios.plot(kind='bar', stacked=False, ax=ax, color=colors)

# 添加百分比标注
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height + 1, f'{height:.1f}%', ha='center', va='bottom')

# 整体卡方检验：三组之间是否有差异
overall_contingency_table = binary_counts.T  # 转置以适应卡方检验的输入
chi2_overall, pvalue_overall, _, _ = stats.chi2_contingency(overall_contingency_table)
# 打印总体差异的 p 值
print(f"Overall p-value comparing all groups: {pvalue_overall:.6f}")

# 如果三组之间有整体差异，继续进行两两比较
if pvalue_overall < 0.05:
    p_values = []
    comparisons = []
    group_pairs = list(itertools.combinations(binary_counts.index, 2))  # 生成所有可能的组对

    for group1, group2 in group_pairs:
        group1_data = binary_counts.loc[group1, :]
        group2_data = binary_counts.loc[group2, :]

        # 卡方检验
        contingency_table = pd.DataFrame([group1_data, group2_data])
        chi2, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        p_values.append(pvalue)
        comparisons.append((group1, group2))

    # 对 p 值进行多重比较校正
    reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

    # 初始化线条位置
    h_step = 10  # 每个横线之间的间距
    h_base = binary_ratios.max().max() + 5  # 最低横线位置
    current_height = h_base

    for (group1, group2), reject in zip(comparisons, reject):
        if reject:  # 只有当校正后的 p 值小于 0.05 时才添加横线和星号
            x1 = binary_counts.index.get_loc(group1)
            x2 = binary_counts.index.get_loc(group2)

            # 每次使用不同的高度来避免重叠
            y = current_height
            current_height += h_step  # 增加高度以备下次使用

            ax.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, color='black')
            ax.text((x1 + x2) / 2, y + 1, '*', ha='center', va='bottom', color='black', fontsize=14)
# 设置图例和轴标签
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['class1','class2' ], title=binary_indicator_column, bbox_to_anchor=(0.9, 1), loc='upper left',
          title_fontsize=14, fontsize=14)
ax.set_xlabel('Cluster', fontsize=16)
ax.set_ylabel('Percentage', fontsize=16)

# 移除上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.subplots_adjust(right=0.9)
# 确保横轴刻度正
plt.xticks(rotation=0, fontsize=16)
plt.yticks(fontsize=16)

plt.show()