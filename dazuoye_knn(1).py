#coding=GBK

import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import SelectKBest, chi2  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, f1_score  
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import RandomUnderSampler

# ���� 1: ��������  
file_path = r"D:\cxdownload\data.csv"  
data = pd.read_csv(file_path)  

# ���� 2: ɾ����һ�в����� 'sex' ��  
data.drop(data.columns[0], axis=1, inplace=True)  
data['Sex'] = (data['Sex'] >= 0.5).astype(int)  # ��ֵת��Ϊ������  

# ���� 3: �� 'sex' �н��ж��ȱ���  
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)  

# ���� 4: ����������Ŀ�����  
X = data.drop(columns=['target'])  # ����  
y = data['target']  # Ŀ�����  

# ���� 5: ʹ�ÿ�������ѡ��ǰ 15 ������  
kbest = SelectKBest(chi2, k=18)  # ���ڿ�������ѡ��ǰ 15 ������  
X_new = kbest.fit_transform(X, y)  

# ��ȡѡ�����������  
selected_features = kbest.get_support(indices=True)  
X_selected = X.iloc[:, selected_features]  # ��ȡ�����Ӽ�  

print("ѡ�����������:", X_selected.columns.tolist())  

# ���� 6: �����ݷ�Ϊѵ�����Ͳ��Լ�  
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)  

# ���� 2: ������������Ƿ���������  
over = SMOTE(sampling_strategy='minority')  # ֻ������������  
under = RandomUnderSampler(sampling_strategy='majority')  # Ƿ����������  
pipeline = Pipeline(steps=[('o', over), ('u', under)])  

# ���� 3: ��ѵ������Ӧ���ز���  
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)  

# �鿴�ز���������ֲ�  
print("�ز���������ֲ���")  
print(pd.Series(y_resampled).value_counts()) 

# ���� 7: ���� KNN �����������ѵ������  
knn = KNeighborsClassifier(n_neighbors=3)  # �ɸ�����Ҫ���� n_neighbors  
knn.fit(X_train, y_train)  

# ���� 8: ����Ԥ��  
y_pred = knn.predict(X_test)  

# ���� 9: ����ģ��  
# ��ӡ��������ͷ��౨��  
print("��������:")  
print(confusion_matrix(y_test, y_pred))  
print("\n���౨��:")  
print(classification_report(y_test, y_pred))  

hhh=knn.predict(X_train)
f11 = f1_score(y_train, hhh, average='macro')  # ���� F1 ����  
print("\nF11 Score (macro):", f11)
# ���㲢��ӡ F1 ����  
f1 = f1_score(y_test, y_pred, average='macro')  # ���� F1 ����  
print("\nF1 Score (macro):", f1)