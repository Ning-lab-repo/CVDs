import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch, Objective

# 读取数据
data = pd.read_csv(r'data.csv',
                   engine='python', encoding='gb18030')

# 提取特征和标签
X = data.drop(['id'], axis=1)
y = data['status']
ID = data['id']

# 划分训练集和测试集，保留对应的 SEQN 号
X_train, X_test, y_train, y_test, ID_train, ID_test = train_test_split(X, y, ID, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 过采样训练集
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# 构建模型的函数，供 Keras Tuner 使用
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=48, max_value=64, step=8),
                    input_dim=X_train_res.shape[1],
                    activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)))
    model.add(Dense(16, activation='relu'))  # 固定倒数第二层为8维
    model.add(Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])

    return model


# 使用 RandomSearch 进行超参数调优
tuner = RandomSearch(build_model,
                     objective=Objective("val_auc", direction="max"),
                     max_trials=5,  # 搜索5种不同的超参数组合
                     executions_per_trial=2,  # 每种组合执行2次
                     directory='my_dir16',
                     project_name='dnn_tuning_fixed_16')

# 设置早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 搜索最优超参数
tuner.search(X_train_res, y_train_res,
             validation_data=(X_test, y_test),
             epochs=100,  # 每个模型训练最多20个epoch
             callbacks=[early_stopping])

# 获得最优模型
best_model = tuner.get_best_models(num_models=1)[0]

# 在测试集上预测概率
y_pred_proba = best_model.predict(X_test)

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DNN (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.subplots_adjust(bottom=0.2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('False positive rate', fontsize=16)
plt.ylabel('True positive rate', fontsize=16)
plt.title('ROC', fontsize=16, pad=10, loc='center')
plt.legend(loc='lower right', fontsize=14)
plt.show()

# 保存模型为.h5文件
best_model.save(r'DNN模型16维_tuned_16_fixed.h5')

print("Best model saved as 'DNN模型.h5'.")

# 提取倒数第二层（第二个 Dense 层）的结果
intermediate_layer_model = Model(inputs=best_model.input, outputs=best_model.layers[-3].output)
intermediate_output_train = intermediate_layer_model.predict(X_train)
intermediate_output_test = intermediate_layer_model.predict(X_test)

# 合并训练集和测试集的中间层输出
intermediate_output = np.vstack((intermediate_output_train, intermediate_output_test))

ID_all = np.concatenate((ID_train.values, ID_test.values))
y_all = np.concatenate((y_train.values, y_test.values))

# 将结果保存到新的CSV文件中，只包含中间层输出和对应的 SEQN 号
intermediate_output_df = pd.DataFrame(intermediate_output,
                                      columns=[f'layer_{i}' for i in range(intermediate_output.shape[1])])
intermediate_output_df['id'] = ID_all
intermediate_output_df['status'] = y_all
intermediate_output_df.to_csv(r'DNN的特征结果16维.csv', index=False)

print("Intermediate layer output saved to 'DNN的特征结果.csv'.")