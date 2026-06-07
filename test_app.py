import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

print("Model eğitiliyor, lütfen bekleyin...")

# 1. Modeli Eğitme
df = pd.read_csv('cs-training.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']
sutun_isimleri = X.columns 

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

lgbm = LGBMClassifier(is_unbalance=True, random_state=42, n_estimators=100, verbose=-1)
lgbm.fit(X_imputed, y)

# 2. Hazır Profiller
profiller = {
    "1": {"isim": "Müşteri A", "veri": [0.10, 45, 0, 0.20, 25000, 5, 0, 1, 0, 2]},
    "2": {"isim": "Müşteri B", "veri": [0.95, 23, 1, 0.60, 17000, 3, 0, 0, 0, 0]},
    "3": {"isim": "Müşteri C", "veri": [0.30, 40, 0, 0.40, 60000, 8, 2, 2, 0, 1]},
    "4": {"isim": "Müşteri D", "veri": [0.05, 68, 0, 0.15, 12500, 2, 0, 0, 0, 0]},
    "5": {"isim": "Müşteri E", "veri": [0.70, 35, 2, 0.45, 22000, 4, 0, 1, 0, 2]},
    "6": {"isim": "Müşteri F", "veri": [0.00, 22, 0, 0.05, 15000, 1, 0, 0, 0, 0]},
    "7": {"isim": "Müşteri G", "veri": [1.00, 38, 3, 0.85, 18000, 6, 4, 0, 2, 3]},
    "8": {"isim": "Müşteri H", "veri": [0.80, 42, 0, 0.55, 45000, 15, 0, 3, 0, 2]},
    "9": {"isim": "Müşteri I", "veri": [0.60, 39, 1, 0.50, 20000, 3, 0, 0, 0, 5]},
    "10": {"isim": "Müşteri J", "veri": [0.01, 55, 0, 0.02, 35000, 4, 0, 1, 0, 0]}
}

def metrikleri_yazdir(veri):
    print(f"  Yaş: {veri[1]} | Aylık Gelir: {veri[4]} TL | Borç/Gelir Oranı: {veri[3]:.2f}")
    print(f"  Gecikmeler -> 30-59 Gün: {veri[2]} | 60-89 Gün: {veri[8]} | 90+ Gün: {veri[6]}")
    print(f"  Limit Doluluk: {veri[0]:.2f} | Açık Kredi Sayısı: {veri[5]} | Konut Kredisi: {veri[7]} | Bağımlı Kişi: {veri[9]}")

# 3. Karar Motoru Fonksiyonu
def risk_hesapla(kisi_verisi):
    kisi_df = pd.DataFrame([kisi_verisi], columns=sutun_isimleri)
    kisi_imputed = imputer.transform(kisi_df)
    
    skor = lgbm.predict_proba(kisi_imputed)[0][1]
    
    print(f"\n[ RİSK SKORU: %{skor*100:.2f} ]")
    if skor < 0.20:
        print("KARAR: ONAY")
    elif skor <= 0.40:
        print("KARAR: İNCELEME")
    else:
        print("KARAR: RET")
    print("-" * 115)

# Yeni Fonksiyon: Proje Raporunu Yazdır
def raporu_yazdir():   
    print(" ***** TEST SONUÇLARI:")
    print("1. Lojistik Regresyon (Baseline Model)")
    print("   - Duyarlılık (Recall): %66.03")
    print("   - Ayrıştırma Gücü (AUC-ROC): %80.32")
    print("   * Dezavantaj: Veri ölçeklendirmesine (Scaling) ihtiyaç duyar, karmaşık senaryolarda zayıftır.\n")
    
    print("2. LightGBM (Geliştirilen Ana Model)")
    print("   - Duyarlılık (Recall): %76.96")
    print("   - Ayrıştırma Gücü (AUC-ROC): %86.72")
    print("   * Avantaj: Temerrüde düşecek (batık) müşterileri tespit etmede Lojistik Regresyon'a göre")
    print("     yaklaşık %11 daha başarılı olmuştur. Finansal risk yönetiminde hayati olan 'Yanlış Negatif'")
    print("     oranını büyük ölçüde düşürmüştür.")
    print("="*115 + "\n")

# 4. İnteraktif Menü Tablosu
while True:
    print("\n" + "="*115)
    print(f"{'No':<3} | {'İsim':<10} | {'Limit':<5} | {'Yaş':<3} | {'30-59G':<6} | {'BorçOr.':<7} | {'Gelir(TL)':<9} | {'AçıkKr.':<7} | {'90+G':<4} | {'KonutKr':<7} | {'60-89G':<6} | {'Bağımlı':<7}")
    print("-" * 115)
    for k, v in profiller.items():
        veri = v['veri']
        print(f"{k:<3} | {v['isim']:<10} | {veri[0]:<5.2f} | {veri[1]:<3} | {veri[2]:<6} | {veri[3]:<7.2f} | {veri[4]:<9} | {veri[5]:<7} | {veri[6]:<4} | {veri[7]:<7} | {veri[8]:<6} | {veri[9]:<7}")
    
    print("-" * 115)
    print(f"{'11':<3} | {'TEST İÇİN MANUEL MÜŞTERİ GİRİŞİ':<106}")
    print(f"{'12':<3} | {'PROJE RAPORU GÖRÜNTÜLE':<106}")
    print(f"{'0':<3} | {'SİSTEMDEN ÇIKIŞ':<106}")
    print("="*115)
    
    secim = input("\nİşlem seçin (0-12): ")
    
    if secim == '0':
        print("Sistem kapatılıyor.")
        break
    elif secim == '12':
        raporu_yazdir()
    elif secim in profiller:
        secilen_profil = profiller[secim]
        print(f"\n--- {secilen_profil['isim'].upper()} METRİKLERİ ---")
        metrikleri_yazdir(secilen_profil['veri'])
        risk_hesapla(secilen_profil['veri'])
    elif secim == '11':
        print("\n--- TEST MÜŞTERİSİ İÇİN METRİKLERİ GİRİN ---")
        try:
            limit = float(input("1. Kredi Limit Doluluk Oranı (Örn 0.50): "))
            yas = float(input("2. Yaş: "))
            gecikme_30 = float(input("3. 30-59 Gün Gecikme Sayısı: "))
            borc_orani = float(input("4. Borç/Gelir Oranı (Örn 0.30): "))
            gelir = float(input("5. Aylık Gelir (TL): "))
            acik_kredi = float(input("6. Açık Kredi/Kart Sayısı: "))
            gecikme_90 = float(input("7. 90+ Gün Gecikme Sayısı: "))
            konut_kredi = float(input("8. İpotekli Konut Kredisi Sayısı: "))
            gecikme_60 = float(input("9. 60-89 Gün Gecikme Sayısı: "))
            bagimli = float(input("10. Bakmakla Yükümlü Olunan Kişi Sayısı: "))
            
            manuel_veri = [limit, yas, gecikme_30, borc_orani, gelir, acik_kredi, gecikme_90, konut_kredi, gecikme_60, bagimli]
            print("\n***MANUEL PROFİL METRİKLERİ***")
            metrikleri_yazdir(manuel_veri)
            risk_hesapla(manuel_veri)
        except ValueError:
            print("Veri giriş hatası. Yalnızca sayısal değerler geçerlidir.")
    else:
        print("Geçersiz seçim!")