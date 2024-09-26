import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import os
import matplotlib.ticker as ticker


# 读取CSV文件
data = pd.read_csv(r'data.csv')

# 确保 Cluster 列的值是整数，并且没有不可见字符
data['Cluster'] = data['Cluster'].astype(str).str.strip()

# 要排除的列名
exclude_columns = ['id'等]
group_column = 'Cluster'

# 需要绘制的列名
columns_to_plot = [col for col in data.columns if col not in exclude_columns]

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'

# 设置离群值的样式
flierprops = dict(marker='o', color='black', markersize=5, markerfacecolor='black')

# 定义颜色
colors = ['#2ECC71', '#4DBAD6', '#E44A33']

# 创建保存图表的文件夹
output_dir = r'箱型图'
os.makedirs(output_dir, exist_ok=True)

for value_column in columns_to_plot:

    # 去除空白值
    data_clean = data[[group_column, value_column]].dropna()
    # 清理数据，将 value_column 转换为数值类型，并过滤掉无法转换的行
    data_clean[value_column] = pd.to_numeric(data_clean[value_column], errors='coerce')
    data_clean = data_clean.dropna(subset=[value_column])

    # 获取各组的数据
    group1_data = data_clean[data_clean[group_column] == '1'][value_column]
    group2_data = data_clean[data_clean[group_column] == '2'][value_column]
    group3_data = data_clean[data_clean[group_column] == '3'][value_column]

    # 选择ANOVA或非参数检验
    use_anova = True
    if (stats.shapiro(group1_data).pvalue <= 0.05 or
        stats.shapiro(group2_data).pvalue <= 0.05 or
        stats.shapiro(group3_data).pvalue <= 0.05 or
        stats.levene(group1_data, group2_data, group3_data).pvalue <= 0.05):
        use_anova = False

    if use_anova:
        # 进行ANOVA检验
        f_stat, p_value = stats.f_oneway(group1_data, group2_data, group3_data)
        method = "ANOVA"
    else:
        # 进行Kruskal-Wallis检验
        h_stat, p_value = stats.kruskal(group1_data, group2_data, group3_data)
        method = "Kruskal-Wallis"

    # 输出总检验结果
    if p_value < 0.05:
        print(f'{value_column}总体差异显著 ({method})，p-value = {p_value}')
    else:
        print(f'{value_column}总体差异不显著 ({method})，p-value = {p_value}')

    # 如果总体差异显著，进行两两比较并校正p值
    if p_value < 0.05:
        if use_anova:
            tukey = pairwise_tukeyhsd(endog=data_clean[value_column], groups=data_clean[group_column], alpha=0.05)
            pvals = tukey.pvalues
        else:
            pvals = []
            for pair in [(group1_data, group2_data), (group1_data, group3_data), (group2_data, group3_data)]:
                _, p = stats.mannwhitneyu(*pair)
                pvals.append(p)
            pvals = np.array(pvals)

        # 校正p值（多重比较校正）
        _, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')

    # 绘制箱型图
    plt.figure()
    boxplot = data_clean.boxplot(column=value_column, by=group_column, patch_artist=True, return_type='dict',
                                 flierprops=flierprops, showfliers=False)

    # 设置箱型图的颜色和边框颜色
    for patch, color in zip(boxplot[value_column]['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    # 设置中位数线条和极端值线条的颜色
    for median in boxplot[value_column]['medians']:
        median.set(color='black', linewidth=1.5)
    for whisker in boxplot[value_column]['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in boxplot[value_column]['caps']:
        cap.set(color='black', linewidth=1.5)

    # 设置横纵轴标签及其字体大小
    plt.xlabel('Cluster', fontsize=16)
    plt.ylabel(value_column, fontsize=16)

    # 计算不显示离群值的最大值和最小值
    min_without_outliers = data_clean.groupby(group_column)[value_column].apply(
        lambda x: x[x >= (np.percentile(x, 25) - 1.5 * (np.percentile(x, 75) - np.percentile(x, 25)))].min()).min()
    max_without_outliers = data_clean.groupby(group_column)[value_column].apply(
        lambda x: x[x <= (np.percentile(x, 75) + 1.5 * (np.percentile(x, 75) - np.percentile(x, 25)))].max()).max()

    # 设置y轴从0开始，并设置y轴的最大值为不显示离群值的最大值加20%
    y_min=min_without_outliers
    y_max = max_without_outliers * 1.2 # 根据需要调整比例
    # 设置y轴从0开始，并设置y轴的最大值为不显示离群值的最大值加20%
    y_min = min_without_outliers * 0.9
    y_max = max_without_outliers * 1.2  # 根据需要调整比例

    # 手动设置Y轴的刻度，使得0到最小值为一格，其他为等间距
    ax = plt.gca()
    ax.set_ylim(y_min, y_max)
    # 显示x轴和y轴刻度
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(left=0.2)

    # 移除背景线
    plt.grid(False)

    # 移除标题和图例
    plt.title('')
    plt.suptitle('')

    # 移除上方和右方的线框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 如果总体差异显著，添加显著性星号
    if p_value < 0.05:
        h_step = 2  # 将垂直间距减小
        h_base = max_without_outliers * 1.05  # 使基准线更加靠近箱线图的最大值
        comparison_count = 0

        for i in range(len(data_clean[group_column].unique())):
            for j in range(i+1, len(data_clean[group_column].unique())):
                if pvals_corrected[comparison_count] < 0.05:
                    y = h_base + comparison_count * h_step * 4
                    x1 = i + 1
                    x2 = j + 1
                    plt.plot([x1, x1, x2, x2], [y, y + h_step, y + h_step, y], lw=1.5, color='black')
                    plt.plot([x1, x1], [y, y + h_step], lw=1.5, color='black')
                    plt.plot([x2, x2], [y, y + h_step], lw=1.5, color='black')
                    plt.text((x1 + x2) * .5, y + h_step, '*', ha='center', va='bottom', color='black', fontsize=14)

                comparison_count += 1

    # 保存图表到文件夹
    plt.savefig(os.path.join(output_dir, f'{value_column}_boxplot.pdf'))
    plt.close()