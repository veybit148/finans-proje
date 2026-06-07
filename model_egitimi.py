import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer


# 1. Veriyi Yükle ve Hazırla
df = pd.read_csv('cs-training.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# Eksik verileri medyan ile doldur
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Veriyi Eğitim ve Test olarak ayır (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# 2. Modelleri Tanımla ve Eğit
# Lojistik Regresyon (Baseline) - class_weight dengesizliği yönetmek için
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# LightGBM (Ana Model)
lgbm = LGBMClassifier(is_unbalance=True, random_state=42, n_estimators=100, verbose=-1)
lgbm.fit(X_train, y_train)

# 3. Tahminler ve Metrikler
def print_metrics(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n--- {model_name} PERFORMANSI ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print_metrics("LOJİSTİK REGRESYON", log_reg, X_test, y_test)
print_metrics("LIGHTGBM", lgbm, X_test, y_test)