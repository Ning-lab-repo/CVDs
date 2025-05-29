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
data = pd.read_csv(r'df.csv', engine='python')

# 提取特征和标签
X = data.drop(['ID"], axis=1)
y = data['status']
ID = data['ID']

# 划分训练集（70%）和测试集（30%），保留对应的 ID
X_train_val, X_test, y_train_val, y_test, ID_train_val, ID_test = train_test_split(X, y, ID, test_size=0.3, random_state=42, stratify=y)

# 再将训练集划分为训练集（80%）和验证集（20%），用于超参数调优
X_train, X_val, y_train, y_val, ID_train, ID_val = train_test_split(X_train_val, y_train_val, ID_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# **1. 读取外部验证数据**
#external_data = pd.read_csv(r'X.csv')

# **2. 提取特征和 ID**
#X_external = external_data.drop(['ID'], axis=1)
#ID_external = external_data['ID']
#y_external = external_data['status']
# 标准化数据（使用训练集的均值和标准差）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 过采样训练集（仅对训练集进行 SMOTE 处理）
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 构建模型的函数，供 Keras Tuner 使用
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=1, max_value=64, step=16),
                    input_dim=X_train_res.shape[1],
                    activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)))
    model.add(Dense(8, activation='relu'))  # 固定倒数第二层为8维
    model.add(Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC'])

    return model

# 使用 RandomSearch 进行超参数调优（在训练集和验证集上）
tuner = RandomSearch(build_model,
                     objective=Objective("val_auc", direction="max"),
                     max_trials=5,  # 搜索3种不同的超参数组合
                     executions_per_trial=5,  # 每种组合执行2次
                     directory='X',
                     project_name='X')

# 设置早停回调
early_stopping = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)

# 搜索最优超参数
tuner.search(X_train_res, y_train_res,
             validation_data=(X_val, y_val),  # 现在用验证集选择超参数
             epochs=20,
             callbacks=[early_stopping])

# 获得最优模型
best_model = tuner.get_best_models(num_models=1)[0]

# 在测试集上预测概率（最终评估）
y_pred_proba = best_model.predict(X_test)


# 保存模型
best_model.save(r'model.h5')

print("Best model saved as 'DNN模型.h5'.")

# 提取倒数第二层的输出
intermediate_layer_model = Model(inputs=best_model.input, outputs=best_model.layers[X].output)
# 合并对应的 ID 和状态（status）
ID_all = np.concatenate((ID_train.values, ID_val.values, ID_test.values))
y_all = np.concatenate((y_train.values, y_val.values, y_test.values))

# 保存为 DataFrame
intermediate_output_df = pd.DataFrame(intermediate_output,
                                      columns=[f'layer_{i}' for i in range(intermediate_output.shape[1])])
intermediate_output_df['ID'] = ID_all
intermediate_output_df['status'] = y_all

# 保存 CSV 文件
intermediate_output_df.to_csv(r'X.csv', index=False)
