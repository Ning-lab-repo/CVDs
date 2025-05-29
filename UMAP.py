import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载CSV文件
data = pd.read_csv(r'X.csv', engine='python', encoding='gb18030')

# 指定要用于UMAP降维的特征列（去除 ID 和 cluster 信息）
features = [col for col in data.columns if col not in ['X']]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# 执行 UMAP 降维
umap_model = umap.UMAP(n_components=2, random_state=42)
data_umap = umap_model.fit_transform(data_scaled)

# 将UMAP结果添加回数据框
data['UMAP1'] = data_umap[:, 0]
data['UMAP2'] = data_umap[:, 1]

# 创建映射字典，将 cluster 映射到英文标签
cluster_labels = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}

# 获取数据中唯一的 cluster 值并排序
unique_clusters = sorted(data['cluster'].unique())

# ✅ 手动指定颜色（浅色）：绿色、蓝色、红色
cluster_color_mapping = {
    1: '#2ca02c',
    2: '#1f77b4',
    3: '#e84a4a'
}

# 设置字体
plt.rcParams['font.sans-serif'] = 'Arial'

# 绘制UMAP结果
plt.figure(figsize=(8, 8))
for cluster in unique_clusters:
    plt.scatter(data[data['cluster'] == cluster]['UMAP1'],
                data[data['cluster'] == cluster]['UMAP2'],
                color=cluster_color_mapping[cluster],
                label=cluster_labels.get(cluster, f'Cluster {cluster}'),
                s=10,
                alpha=0.7)

plt.xlabel('UMAP 2D component 1', fontsize=28)
plt.ylabel('UMAP 2D component 2', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.subplots_adjust(right=0.8)
# plt.title('UMAP', fontsize=28)
plt.legend(title='X', bbox_to_anchor=(1, 1), loc='upper left', fontsize=18, title_fontsize=18)
plt.show()

