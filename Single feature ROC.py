import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 读取数据
data = pd.read_csv(r'data.csv',
                   engine='python', encoding='gb18030')

# 假设 y_true 是三分类标签
y_true = data['results']

# 将多分类标签二值化
y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])

# 示例指标
scores_list = [
    ('feature1', data['feature1']),
    ('feature2', data['feature2']),
    ('feature3', data['feature3'])
]

# 存储各指标的ROC和AUC
roc_auc_dict = {}

# 使用 OneVsRestClassifier 将每个特征单独分类
for name, scores in scores_list:
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(scores.values.reshape(-1, 1), y_true)
    y_score = classifier.decision_function(scores.values.reshape(-1, 1))

    # 对每个类别计算 ROC
    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[name] = roc_auc
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

# 根据AUC值进行排序
roc_auc_sorted = dict(sorted(roc_auc_dict.items(), key=lambda item: item[1], reverse=True))

# 绘制按AUC排序的ROC曲线
plt.figure()
for name, auc_value in roc_auc_sorted.items():
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(data[name].values.reshape(-1, 1), y_true)
    y_score = classifier.decision_function(data[name].values.reshape(-1, 1))

    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_value:.4f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.rcParams['font.sans-serif'] = 'Arial'
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.subplots_adjust(bottom=0.2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('False positive rate', fontsize=16)
plt.ylabel('True positive rate', fontsize=16)
plt.title('ROC for different features', fontsize=16, pad=10, loc='center')
plt.legend(loc='lower right', fontsize=8)
plt.show()
