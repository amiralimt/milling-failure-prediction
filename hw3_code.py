import numpy as np #for data processing - linear algebra
import pandas as pd #for data processing - csv file
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization
from matplotlib.table import table

from sklearn.model_selection import train_test_split #for machine learning models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score ,roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier  # برای SVM چندکلاسه
from sklearn.preprocessing import label_binarize

import os #for file managment
from imblearn.over_sampling import SMOTE

#تنظیم نمایش کامل جدول‌ها
pd.set_option('display.max_columns', None)  # نشون دادن همه ستون‌ها
pd.set_option('display.width', 1000)        # عرض جدول در خروجی
pd.set_option('display.max_colwidth', None) # نمایش کامل مقادیر داخل سلول‌ها (برای ستون‌های متنی)
pd.set_option('display.max_rows', None)

#A1---------------------------------------
print("⁉️ بخش الف - ۱:")
dataset = pd.read_csv('milling_machine.csv')
print(dataset.head(10))
dataset.info()
print(dataset.describe())

#A2----------------------------------------
print("⁉️ بخش الف - ۲:")
missing_count = dataset.isnull().sum()
missing_ratio = (missing_count / len(dataset)) *100
missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Ratio (%)': missing_ratio
})
print(missing_df)

#A3---------------------------------------------
print("⁉️ بخش الف - ۳:")
# تبدیل Failure Types به مقادیر عددی برای بررسی همبستگی
dataset['Failure Types (Encoded)'] = dataset['Failure Types'].astype('category').cat.codes
correlation_matrix = dataset.corr(numeric_only=True)

print(correlation_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#A4------------------------------------------------------------
print("⁉️ بخش الف - ۴:")
important_features = ['Tool Wear (Seconds)', 'Air Temp (°C)', 'Torque (Nm)']

for feature in important_features:
    plt.figure(figsize=(10,6))
    sns.histplot(dataset[feature], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

#B1---------------------------------------------------------
print("⁉️ بخش ب - ۱:")
df = dataset.copy()

# لیست ویژگی‌های عددی (بدون ستون Failure Types و بدون نسخه Encode شده)
features = ['Air Temp (°C)', 'Process Temp (°C)', 'Rotational Speed (RPM)', 'Torque (Nm)', 'Tool Wear (Seconds)']
classes = df['Failure Types'].unique()

# ساخت جدول میانگین هر ویژگی بر اساس کلاس
mean_class = pd.DataFrame(index=features, columns=classes)

# محاسبه میانگین برای هر ویژگی در هر کلاس
for f in classes:
    class_data = df[df['Failure Types'] == f]
    mean_values = class_data[features].mean()
    mean_class.loc[:, f] = mean_values

# پیدا کردن موقعیت مقادیر گمشده
nan_locations = []
for index, row in df.iterrows():
    for col in features:
        if pd.isnull(row[col]):
            nan_locations.append((index, col))

# جایگزینی مقادیر گمشده با میانگین همان کلاس
for (index, column) in nan_locations:
    class_label = df.loc[index, 'Failure Types']
    df.loc[index, column] = mean_class.loc[column, class_label]

# جایگزین کردن دیتاست اصلی با نسخه پاکسازی‌شده
dataset = df.copy()
print(dataset.isnull().sum())
# حذف ردیف‌هایی که Failure Types ناموجود دارن
dataset = dataset.dropna(subset=['Failure Types']).reset_index(drop=True)
print("\n number of missing values in failure type: " + str(dataset['Failure Types'].isnull().sum()))

#B2--------------------------------------------------------------
print("⁉️ بخش ب - ۲:")
features_to_scale = ['Air Temp (°C)', 'Process Temp (°C)', 'Rotational Speed (RPM)', 'Torque (Nm)', 'Tool Wear (Seconds)']
dataset_scaled = dataset.copy()

scaler = StandardScaler()
dataset_scaled[features_to_scale] = scaler.fit_transform(dataset_scaled[features_to_scale])

print(dataset_scaled[features_to_scale].describe())

# ساخت ستون انکد شده نهایی از Failure Types برای مدل‌سازی
dataset_scaled['Failure Types (Encoded)'] = dataset_scaled['Failure Types'].astype('category').cat.codes

dataset_scaled.info()

#j1---------------------------------------------------------------
print("⁉️ بخش ج - ۱:")
# ساخت ستون دودسته‌ای جدید به نام 'Binary Failure'
dataset_scaled['Binary Failure'] = dataset_scaled['Failure Types'].apply(
    lambda x: 0 if x == 'No Failure' else 1
)
print(dataset_scaled[['Failure Types', 'Binary Failure']].head(10))
print(dataset_scaled['Binary Failure'].value_counts())

#j2---------------------------------------------------------------
print("⁉️ بخش ج - ۲:")
sns.countplot(x='Binary Failure', hue='Binary Failure', data=dataset_scaled, palette='deep', legend=False)
plt.title('Distribution of Binary Failure Classes')
plt.xlabel('Failure Class (0 = No Failure, 1 = Failure)')
plt.ylabel('Count')
plt.grid(True, axis='y')
plt.show()

#j3-----------------------------------------------------------------
print("⁉️ بخش ج - ۳:")
print(dataset_scaled['Binary Failure'].value_counts(normalize=True))

#j4-----------------------------------------------------------------
print("⁉️ بخش ج - ۴:")
X = dataset_scaled[features_to_scale]
y = dataset_scaled['Binary Failure']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_resampled_df = pd.DataFrame(X_resampled, columns=features_to_scale)
y_resampled_df = pd.DataFrame(y_resampled, columns=['Binary Failure'])

print(y_resampled_df['Binary Failure'].value_counts())

#j5---------------------------------------------------------
print("⁉️ بخش ج - ۵:")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_df,  # ویژگی‌ها
    y_resampled_df,  # برچسب‌ها
    test_size=0.2,
    random_state=42,
    stratify=y_resampled_df  # برای حفظ نسبت کلاس‌ها
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train.values.ravel())

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train.values.ravel())

# SVM - Linear kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train.values.ravel())

# SVM - RBF kernel (غیرخطی)
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train.values.ravel())

#j6-------------------------------------------------------------------
print("⁉️ بخش ج - ۶:")
models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "SVM (Linear)": svm_linear,
    "SVM (RBF)": svm_rbf
}
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nModel: {name}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision (class 1)": round(report['1']['precision'], 4),
        "Recall (class 1)": round(report['1']['recall'], 4),
        "F1-score (class 1)": round(report['1']['f1-score'], 4)
    })

results_df = pd.DataFrame(results)
print("\n\n📊 جدول مقایسه‌ای مدل‌ها:\n")
print(results_df)

#j7---------------------------------------------------------------------
print("⁉️ بخش ج - ۷:")
knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11]
}

grid_search_knn = GridSearchCV(knn, param_grid_knn, scoring='accuracy', cv=5)
grid_search_knn.fit(X_train, y_train.values.ravel())

print("KNN - Best K:", grid_search_knn.best_params_)
print("Best Accuracy:", round(grid_search_knn.best_score_, 4))

best_knn = grid_search_knn.best_estimator_

log_reg = LogisticRegression(solver='liblinear', random_state=42)

param_grid_log = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

grid_search_log = GridSearchCV(log_reg, param_grid_log, scoring='accuracy', cv=5)
grid_search_log.fit(X_train, y_train.values.ravel())

print("Logistic Regression - Best Params:", grid_search_log.best_params_)
print("Best Accuracy:", round(grid_search_log.best_score_, 4))

best_log_model = grid_search_log.best_estimator_

svm_linear = SVC(kernel='linear', random_state=42)

param_grid_linear = {
    'C': [0.01, 0.1, 1, 10]
}

grid_search_linear = GridSearchCV(svm_linear, param_grid_linear, scoring='accuracy', cv=5)
grid_search_linear.fit(X_train, y_train.values.ravel())

print("SVM (Linear) - Best Params:", grid_search_linear.best_params_)
print("Best Accuracy:", round(grid_search_linear.best_score_, 4))

best_svm_linear = grid_search_linear.best_estimator_

svm_rbf = SVC(kernel='rbf', random_state=42)

param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}

grid_search_rbf = GridSearchCV(svm_rbf, param_grid_rbf, scoring='accuracy', cv=5)
grid_search_rbf.fit(X_train, y_train.values.ravel())

print("SVM (RBF) - Best Params:", grid_search_rbf.best_params_)
print("Best Accuracy:", round(grid_search_rbf.best_score_, 4))

best_svm_rbf = grid_search_rbf.best_estimator_

#j8------------------------------------------------------------------------
print("⁉️ بخش ج - ۸:")
optimized_models = {
    "KNN (Optimized)": best_knn,
    "Logistic Regression (Optimized)": best_log_model,
    "SVM (Linear, Optimized)": best_svm_linear,
    "SVM (RBF, Optimized)": best_svm_rbf
}
metrics_results = []

for name, model in optimized_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    err = 1 - acc
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix: [TN, FP, FN, TP]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # AUC only if model supports predict_proba
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    metrics_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Error Rate": round(err, 4),
        "Precision (class 1)": round(precision, 4),
        "Recall (class 1)": round(recall, 4),
        "Specificity (class 0)": round(specificity, 4),
        "F1-score (class 1)": round(f1, 4),
        "AUC-ROC": round(auc, 4) if auc != "N/A" else "N/A",
        "Confusion Matrix": f"{cm.tolist()}"
    })

metrics_df = pd.DataFrame(metrics_results)
pd.set_option("display.max_colwidth", None)
print("📊 جدول نهایی مقایسه مدل‌ها با تمام شاخص‌ها:\n")
print(metrics_df)

#d1---------------------------------------------------------------------------
print("⁉️ بخش د - ۱:")
# X و y برای دسته‌بندی چندکلاسه
X_multi = dataset_scaled[features_to_scale]
y_multi = dataset_scaled['Failure Types (Encoded)']

X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi)

# 1. KNN
knn_mc = KNeighborsClassifier(n_neighbors=5)
knn_mc.fit(X_train_mc, y_train_mc)

# 2. Decision Tree
dt_mc = DecisionTreeClassifier(random_state=42)
dt_mc.fit(X_train_mc, y_train_mc)

# 3. Random Forest
rf_mc = RandomForestClassifier(n_estimators=100, random_state=42)
rf_mc.fit(X_train_mc, y_train_mc)

# 4. SVM با One-vs-Rest (برای چندکلاسه)
svm_mc = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))
svm_mc.fit(X_train_mc, y_train_mc)

#d2-----------------------------------------------------------------------
print("⁉️ بخش د - ۲:")

multi_models = {
    "KNN": knn_mc,
    "Decision Tree": dt_mc,
    "Random Forest": rf_mc,
    "SVM (OvR)": svm_mc
}
multi_results = []

for name, model in multi_models.items():
    y_pred = model.predict(X_test_mc)

    acc = accuracy_score(y_test_mc, y_pred)
    cm = confusion_matrix(y_test_mc, y_pred)
    report = classification_report(y_test_mc, y_pred, output_dict=True, zero_division=0)

    print(f"\n🔹 {name}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test_mc, y_pred, zero_division=0))

    # استخراج میانگین macro برای مقایسه کلی
    multi_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Macro F1": round(report['macro avg']['f1-score'], 4),
        "Macro Precision": round(report['macro avg']['precision'], 4),
        "Macro Recall": round(report['macro avg']['recall'], 4),
        "Confusion Matrix": cm.tolist()
    })

results_df_mc = pd.DataFrame(multi_results)
pd.set_option("display.max_colwidth", None)
print("\n📊 جدول مقایسه‌ای عملکرد مدل‌های چندکلاسه:")
print(results_df_mc)

#d3--------------------------------------------------------------------------
print("⁉️ بخش د - ۳:")

knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11]
}

grid_knn = GridSearchCV(knn, param_grid_knn, scoring='accuracy', cv=5)
grid_knn.fit(X_train_mc, y_train_mc)

print("KNN - Best Params:", grid_knn.best_params_)
print("Best Accuracy:", round(grid_knn.best_score_, 4))
best_knn_mc = grid_knn.best_estimator_


dt = DecisionTreeClassifier(random_state=42)

param_grid_dt = {
    'max_depth': [5, 10, 20, None],
    'criterion': ['gini', 'entropy']
}

grid_dt = GridSearchCV(dt, param_grid_dt, scoring='accuracy', cv=5)
grid_dt.fit(X_train_mc, y_train_mc)

print("Decision Tree - Best Params:", grid_dt.best_params_)
print("Best Accuracy:", round(grid_dt.best_score_, 4))
best_dt_mc = grid_dt.best_estimator_


rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None]
}

grid_rf = GridSearchCV(rf, param_grid_rf, scoring='accuracy', cv=5)
grid_rf.fit(X_train_mc, y_train_mc)

print("Random Forest - Best Params:", grid_rf.best_params_)
print("Best Accuracy:", round(grid_rf.best_score_, 4))
best_rf_mc = grid_rf.best_estimator_


svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))

param_grid_svm = {
    'estimator__C': [0.1, 1, 10],
    'estimator__gamma': [0.01, 0.1, 1]
}

grid_svm = GridSearchCV(svm, param_grid_svm, scoring='accuracy', cv=5)
grid_svm.fit(X_train_mc, y_train_mc)

print("SVM (OvR) - Best Params:", grid_svm.best_params_)
print("Best Accuracy:", round(grid_svm.best_score_, 4))
best_svm_mc = grid_svm.best_estimator_

#d4----------------------------------------------------------------------------
print("⁉️ بخش د - ۴:")
final_multi_models = {
    "KNN (opt)": best_knn_mc,
    "Decision Tree (opt)": best_dt_mc,
    "Random Forest (opt)": best_rf_mc,
    "SVM (OvR, opt)": best_svm_mc
}

# برچسب‌ها به صورت one-hot برای AUC
classes = np.unique(y_test_mc)
y_test_binarized = label_binarize(y_test_mc, classes=classes)

final_multi_results = []

for name, model in final_multi_models.items():
    y_pred = model.predict(X_test_mc)

    # Accuracy & Error
    acc = accuracy_score(y_test_mc, y_pred)
    err = 1 - acc

    # Precision, Recall, F1 (macro)
    precision = precision_score(y_test_mc, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_mc, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_mc, y_pred, average='macro', zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_test_mc, y_pred)

    # Specificity (میانگین کلی از هر کلاس، مثل sensitivity اما برای کلاس منفی)
    # برای چندکلاسه تعریف‌شده به‌صورت TN / (TN + FP)
    specificity_list = []
    for i in range(len(classes)):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_list.append(specificity)
    specificity_avg = np.mean(specificity_list)

    # AUC-ROC (در صورت وجود احتمال)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_mc)
        auc = roc_auc_score(y_test_binarized, y_score, multi_class='ovr', average='macro')
    else:
        auc = "N/A"

    final_multi_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Error Rate": round(err, 4),
        "Precision (Macro)": round(precision, 4),
        "Recall (Macro)": round(recall, 4),
        "F1-score (Macro)": round(f1, 4),
        "Specificity (Macro)": round(specificity_avg, 4),
        "AUC-ROC": round(auc, 4) if auc != "N/A" else "N/A",
        "Confusion Matrix": cm.tolist()
    })

df_d4 = pd.DataFrame(final_multi_results)
pd.set_option('display.max_colwidth', None)
print("📊 جدول نهایی مقایسه مدل‌های چندکلاسه بهینه‌شده:")
print(df_d4)