import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. GÉNÉRATION DU FICHIER DE TEST (30 JOURS)
# ==========================================
def generer_donnees_test():
    jours = [f'Jour_{i}' for i in range(1, 31)]
    produits = ['AUGMENTIN ADULTE', 'PARACETAMOL 1G', 'VENTOLINE', 'SPASFON', 'AMLO 5MG']
    data = {'DESIGNATION': produits}

    for j in jours:
        # On simule une tendance légèrement haussière pour tester l'IA
        data[j] = np.random.randint(5, 15, size=5) + np.array([0, 1, 2, 3, 4])

    df = pd.DataFrame(data)
    # Stocks critiques pour certains pour déclencher des alertes
    df['STOCK_ACTUEL'] = [50, 200, 15, 80, 10] 
    
    filename = 'test_pharmacie_30j.csv'
    df.to_csv(filename, index=False)
    print(f"✅ Fichier '{filename}' généré pour l'analyse.\n")
    return filename

# ==========================================
# 2. LE CERVEAU IA (TENSORFLOW LSTM)
# ==========================================
def predire_prochaine_vente(historique_30j):
    # On transforme les 30 jours en coordonnées X
    X = np.array(range(len(historique_30j))).reshape(-1, 1)
    y = np.array(historique_30j)
    
    # On utilise une régression polynomiale (elle détecte les courbes/accélérations)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédit le jour 31
    X_31 = poly.transform([[31]])
    prediction = model.predict(X_31)[0]
    
    return max(0, prediction)

# ==========================================
# 3. SCRIPT PRINCIPAL D'ANALYSE
# ==========================================
def main():
    # Générer le fichier si absent
    file_path = generer_donnees_test()
    
    # Lire les données
    df = pd.read_csv(file_path)
    cols_jours = [f'Jour_{i}' for i in range(1, 31)]
    
    print("🚀 Lancement de l'analyse prédictive (TensorFlow LSTM)...")
    print("-" * 70)
    print(f"{'PRODUIT':<20} | {'STOCK':<6} | {'PRÉD. IA':<10} | {'JOURS REST.'}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        nom = row['DESIGNATION']
        stock = row['STOCK_ACTUEL']
        historique = row[cols_jours].values.astype(float)
        
        # Calcul par l'IA
        prediction = predire_prochaine_vente(historique)
        
        # Calcul du temps restant (Stock / Prédiction de demain)
        jours_restants = stock / prediction if prediction > 0.1 else 99
        
        # Formatage de l'alerte
        alerte = ""
        if jours_restants < 5:
            alerte = "⚠️ RUPTURE PROCHE"
        elif jours_restants < 2:
            alerte = "🚨 URGENCE COMMANDE"
            
        print(f"{nom[:20]:<20} | {stock:<6} | {prediction:<10.2f} | {int(jours_restants):<3} jours {alerte}")

if __name__ == "__main__":
    main()
