import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt

# 加载主数据文件
data = pd.read_csv(r'降维使用的数据.csv', engine='python', encoding='gb18030')

# 加载含有 tyg 值的数据文件
tyg_data = pd.read_csv(r'要映射的值的数据.csv', engine='python', encoding='gb18030')

# 根据 SEQN 列合并两个数据框
merged_data = pd.merge(data, tyg_data[['id', 'feature']], on='id', how='left')

# 指定要用于UMAP降维的特征列（除去ID号和分组信息）
features = [col for col in merged_data.columns if col not in ['id', 'cluster', 'feature']]

# 执行UMAP降维
umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.5, spread=1.5)
data_umap = umap_model.fit_transform(merged_data[features])

# 将UMAP结果添加回数据框
merged_data['UMAP1'] = data_umap[:, 0]
merged_data['UMAP2'] = data_umap[:, 1]

# 按照 Predicted_Prob 从小到大排序
merged_data = merged_data.sort_values(by='feature')

# 获取数据中唯一的分组信息
unique_clusters = merged_data['cluster'].unique()

# 生成颜色映射字典，每个分组对应一个颜色
cluster_color_mapping = {cluster: plt.cm.tab10(i) for i, cluster in enumerate(unique_clusters)}
plt.rcParams['font.sans-serif'] = 'Arial'

# 绘制UMAP结果，使用 tyg 值来控制颜色深浅
plt.figure(figsize=(6, 8))
for cluster in unique_clusters:
    subset = merged_data[merged_data['cluster'] == cluster]
    plt.scatter(subset['UMAP1'],
                subset['UMAP2'],
                c=subset['feature'],  # 用 tyg 值控制颜色
                cmap='Blues',   # 使用 'Blues' 色彩映射方案
                label=f'Cluster {cluster}',
                s=30,
                alpha=0.7)

# 在图上标注分组标签
for cluster in unique_clusters:
    subset = merged_data[merged_data['cluster'] == cluster]
    # 计算每个簇中心点位置
    x_mean = subset['UMAP1'].mean()
    y_mean = subset['UMAP2'].mean()
    # 在簇中心位置标注文本
    plt.text(x_mean, y_mean, f'Cluster {cluster}', fontsize=16, ha='center', va='center')

plt.xlabel('UMAP 2D component 1', fontsize=16)
plt.ylabel('UMAP 2D component 2', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-15, 20)
plt.ylim(-5, 20)
plt.subplots_adjust(right=0.9)
plt.title('UMAP of patient clusters with feature value', fontsize=18)

# 添加颜色条并设置颜色条标签的字体大小
cbar = plt.colorbar(label='feature value')
cbar.ax.tick_params(labelsize=16)  # 设置颜色条刻度标签的字体大小
cbar.set_label('feature value', fontsize=14)  # 设置颜色条标签的字体大小

plt.show()

