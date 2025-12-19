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