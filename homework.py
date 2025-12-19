import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Fetching data from UCI Repository...")
try:
    # 1. FETCH DATA
    communities_and_crime = fetch_ucirepo(id=183)
    
    # Extract features (X) and target (y)
    X = communities_and_crime.data.features.copy()
    y = communities_and_crime.data.targets
    
    # --- CLEANING STEP (THE FIX) ---
    print("Cleaning data...")
    
    # 1. Drop text columns that confuse the math (Community Name, State, CountyCode)
    # We drop columns that are 'object' type (text) or completely empty
    X = X.select_dtypes(include=[np.number]) 
    
    # 2. Replace '?' with NaN just in case
    X.replace('?', np.nan, inplace=True)
    
    # 2. PREPROCESSING
    # Convert 'ViolentCrimesPerPop' to Classification (0 or 1)
    y_class = (y['ViolentCrimesPerPop'] > 0.2).astype(int)
    
    # Handle missing values
    # Now that text columns are gone, 'mean' strategy works perfectly
    imputer = SimpleImputer(strategy='mean')
    
    # We use the imputer's output feature names if available, or just index
    X_imputed_data = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_data, columns=X.columns)
    
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    
    # 3. RUN ALGORITHMS
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    results = []
    print("\nRunning Algorithms (this might take 10 seconds)...")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append([name, acc, prec, rec, f1])
        
    # 4. PRINT RESULTS
    results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
    print("\n" + "="*50)
    print("FINAL RESULTS FOR TERM PAPER")
    print("="*50)
    print(results_df)
    print("="*50)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")

# --- WEKA İÇİN DOSYA KAYDETME ---
print("\nWeka için dosya hazırlanıyor...")
# Görselleştirme için ölçeklenmemiş (orijinal) veriyi kullanalım ki rakamlar anlamlı olsun
df_weka = X_imputed.copy()
df_weka['HighCrime'] = y_class  # Hedef sınıfı ekle

# Dosyayı kaydet
df_weka.to_csv("weka_ready.csv", index=False)
print("BAŞARILI! 'weka_ready.csv' dosyası indirilenler klasörüne kaydedildi.")
print("Şimdi bu dosyayı Weka'da sorunsuz açabilirsiniz.")




import matplotlib.pyplot as plt

# --- FEATURE IMPORTANCE (SAYFA DOLDURMAK İÇİN) ---
print("\nÖzellik Önem Düzeyleri Hesaplanıyor...")

# Logistic Regression modelini bul (model listesinin ilk sırasında tanımlamıştık)
log_reg_model = models["Logistic Regression"]

# Katsayıları al
importance = log_reg_model.coef_[0]
feature_names = X.columns

# Pandas serisine çevirip sıralayalım
feat_importances = pd.Series(importance, index=feature_names)
top_features = feat_importances.abs().nlargest(10) # En etkili 10 özellik (Pozitif veya Negatif)

# Grafiği Çiz
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='teal')
plt.title('Top 10 Most Important Factors Predicting High Crime')
plt.xlabel('Impact (Coefficient Value)')
plt.ylabel('Socio-Economic Factors')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Grafik kaydedildi: feature_importance.png")



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- CONFUSION MATRIX ÇİZİMİ ---
print("\nConfusion Matrixler Çiziliyor...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Crime", "High Crime"])
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(f"{name} Confusion Matrix")

plt.tight_layout()
plt.savefig('confusion_matrices.png')
print("Grafik kaydedildi: confusion_matrices.png")
plt.show()
plt.show()