"""
Radiomics Classification + Feature Analysis
Author: doffi
Enhanced Version: with feature exploration, selection, and visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import sys

# --- STEP 1: Load radiomics features (加载特征文件)
if len(sys.argv) > 1:
    dataset_root = os.path.abspath(sys.argv[1])
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(project_root, "dataset", "sample", "CT")

feature_csv = os.path.join(dataset_root, "radiomics_features.csv")
print(f"Loading features from: {feature_csv}")

# --- STEP 2: Prepare features and labels (分离特征与标签)
X = data.drop(columns=["label", "Case"], errors="ignore").select_dtypes(include=["number"])
y = data["label"]

# 如果 label 全相同，自动生成随机标签以便演示
if len(np.unique(y)) == 1:
    print(" Only one class detected, generating demo labels for testing...")
    y = np.random.choice([0, 1], size=len(y))

# --- STEP 3: Feature exploration (特征探索分析)
print("\n Exploring feature correlations...")
corr = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", cbar=False)
plt.title("Feature Correlation Heatmap (特征相关性热图)")
plt.tight_layout()
plt.show()

# --- STEP 4: Feature selection (特征选择)
print("\n Selecting top 10 features (based on ANOVA F-score)...")
selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Top features:", list(selected_features))

# --- STEP 5: Data split & scaling (划分数据集 + 标准化)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 6: Train classifier (训练分类器)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# --- STEP 7: Evaluate model (性能评估)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) > 1 else [0.5] * len(y_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\n Classification Results:")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# --- STEP 8: ROC curve (ROC 曲线)
if len(np.unique(y)) > 1:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate (假阳性率)")
    plt.ylabel("True Positive Rate (真阳性率)")
    plt.title("ROC Curve - Radiomics Classifier")
    plt.legend()
    plt.show()

# --- STEP 9: PCA visualization (PCA 降维可视化)
print("\n Visualizing features with PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_selected)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="coolwarm", s=80)
plt.title("PCA Visualization of Selected Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# --- STEP 10: Cross-validation (交叉验证)
cv = min(3, len(X))  # 防止样本数过少
cv_scores = cross_val_score(clf, X_selected, y, cv=cv)
print(f"\n Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\nRadiomics classification pipeline completed successfully!")
