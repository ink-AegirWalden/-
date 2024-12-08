#coding=GBK

import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import SelectKBest, chi2  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, f1_score  
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import RandomUnderSampler

# 步骤 1: 加载数据  
file_path = r"D:\cxdownload\data.csv"  
data = pd.read_csv(file_path)  

# 步骤 2: 删除第一列并处理 'sex' 列  
data.drop(data.columns[0], axis=1, inplace=True)  
data['Sex'] = (data['Sex'] >= 0.5).astype(int)  # 将值转换为二进制  

# 步骤 3: 对 'sex' 列进行独热编码  
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)  

# 步骤 4: 分离特征和目标变量  
X = data.drop(columns=['target'])  # 特征  
y = data['target']  # 目标变量  

# 步骤 5: 使用卡方检验选择前 15 个特征  
kbest = SelectKBest(chi2, k=18)  # 基于卡方检验选择前 15 个特征  
X_new = kbest.fit_transform(X, y)  

# 获取选择的特征名称  
selected_features = kbest.get_support(indices=True)  
X_selected = X.iloc[:, selected_features]  # 提取特征子集  

print("选择的特征名称:", X_selected.columns.tolist())  

# 步骤 6: 将数据分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)  

# 步骤 2: 创建过采样和欠采样的组合  
over = SMOTE(sampling_strategy='minority')  # 只过采样少数类  
under = RandomUnderSampler(sampling_strategy='majority')  # 欠采样多数类  
pipeline = Pipeline(steps=[('o', over), ('u', under)])  

# 步骤 3: 在训练集上应用重采样  
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)  

# 查看重采样后的类别分布  
print("重采样后的类别分布：")  
print(pd.Series(y_resampled).value_counts()) 

# 步骤 7: 创建 KNN 分类器并拟合训练数据  
knn = KNeighborsClassifier(n_neighbors=3)  # 可根据需要调整 n_neighbors  
knn.fit(X_train, y_train)  

# 步骤 8: 进行预测  
y_pred = knn.predict(X_test)  

# 步骤 9: 评估模型  
# 打印混淆矩阵和分类报告  
print("混淆矩阵:")  
print(confusion_matrix(y_test, y_pred))  
print("\n分类报告:")  
print(classification_report(y_test, y_pred))  

hhh=knn.predict(X_train)
f11 = f1_score(y_train, hhh, average='macro')  # 计算 F1 分数  
print("\nF11 Score (macro):", f11)
# 计算并打印 F1 分数  
f1 = f1_score(y_test, y_pred, average='macro')  # 计算 F1 分数  
print("\nF1 Score (macro):", f1)