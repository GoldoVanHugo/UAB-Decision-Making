"""
Radiomics Classification Example
Author: doffi
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- STEP 1: Load features (加载提取特征CSV文件) ---
feature_csv = r"E:\Projects\UAB-Decision-Making\dataset\sample\CT\radiomics_features.csv"

if not os.path.exists(feature_csv):
    raise FileNotFoundError(f"Feature file not found: {feature_csv}")

data = pd.read_csv(feature_csv)

print("Loaded feature data:", data.shape)
print("Columns:", list(data.columns[:10]), "...")

# --- STEP 2: Separate features and labels (分离特征与标签) ---
# 假设标签列名为 "label"（可以是 0/1）
X = data.drop(columns=["label"])   # Radiomics features
y = data["label"]                  # Target label

# --- STEP 3: Split dataset (划分训练集和测试集) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- STEP 4: Standardize features (特征标准化：z-score标准化) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 5: Train a classifier (训练分类器) ---
# Random Forest 随机森林分类器（Robust and interpretable）
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# --- STEP 6: Predictions (模型预测) ---
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]  # for ROC curve

# --- STEP 7: Evaluate model (性能评估) ---
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Classification Results:")
print(f"Accuracy (准确率): {acc:.3f}")
print(f"AUC (曲线下面积): {auc:.3f}")
print("\nConfusion Matrix (混淆矩阵):\n", cm)
print("\nDetailed Report (详细报告):\n", classification_report(y_test, y_pred))

# --- STEP 8: ROC Curve (ROC曲线绘制) ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate (假阳性率)")
plt.ylabel("True Positive Rate (真阳性率)")
plt.title("ROC Curve - Radiomics Classifier")
plt.legend()
plt.show()

# --- STEP 9: Cross-validation (交叉验证性能) ---
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"\nCross-validation mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
