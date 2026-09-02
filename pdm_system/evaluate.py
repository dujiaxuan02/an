"""
模型评估脚本：读取test_dataset.csv，输出各类指标、混淆矩阵
面向设备预测性维护的智能分析与预警系统
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, f1_score, recall_score, precision_score
)

TEST_CSV = "./data/test_dataset.csv"
MODEL_PATH = "./model/rf_best_model.pkl"

# 加载模型
model = joblib.load(MODEL_PATH)
print(f"成功加载模型：{MODEL_PATH}")

# 加载测试集
df_test = pd.read_csv(TEST_CSV)
drop_cols = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
X_test = df_test.drop(columns=drop_cols)
y_test = df_test["Machine failure"]

# 预测
y_pred = model.predict(X_test)

# 指标计算
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("="*60)
print("====设备故障预测 测试集评估结果====")
print(f"准确率 Accuracy:  {acc:.4f}")
print(f"精确率 Precision:{prec:.4f}")
print(f"召回率 Recall:    {rec:.4f}")
print(f"F1‑score:        {f1:.4f}")
print("="*60)
print("混淆矩阵：")
print(cm)
print("="*60)
print(classification_report(y_test, y_pred))
