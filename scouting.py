import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Adım 1: CSV dosyalarını okutma
attributes_df = pd.read_csv('scoutium_attributes.csv', sep=';')
potential_labels_df = pd.read_csv('scoutium_potential_labels.csv', sep=';')

# Adım 2: Dosyaları birleştirme
merged_df = pd.merge(attributes_df, potential_labels_df, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'])

# Adım 3: Kaleci sınıfını kaldırma
merged_df = merged_df[merged_df['position_id'] != 1]

# Adım 4: below_average sınıfını kaldırma
merged_df = merged_df[merged_df['potential_label'] != 'below_average']

# Adım 5: Pivot table oluşturma
pivot_table = merged_df.pivot_table(index=['player_id', 'position_id', 'potential_label'],
                                  columns='attribute_id',
                                  values='attribute_value',
                                  aggfunc='mean').reset_index()

# Adım 6: Label Encoding
le = LabelEncoder()
pivot_table['potential_label'] = le.fit_transform(pivot_table['potential_label'])

# Adım 7: Sayısal değişken kolonlarını listeye atama
num_cols = pivot_table.columns[3:]

# Adım 8: Veriyi ölçeklendirme
scaler = StandardScaler()
pivot_table[num_cols] = scaler.fit_transform(pivot_table[num_cols])

# Adım 9: Makine öğrenmesi modeli geliştirme
X = pivot_table.drop(['potential_label'], axis=1)
y = pivot_table['potential_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrikleri yazdırma
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Adım 10: Özelliklerin önem düzeyini çizdirme
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()