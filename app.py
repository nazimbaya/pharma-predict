import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

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
    # Normalisation des données entre 0 et 1 (Essentiel pour LSTM)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(historique_30j).reshape(-1, 1))
    
    # Transformation en "Cube" pour TensorFlow : [1 échantillon, 30 timesteps, 1 feature]
    X_input = data_scaled.reshape(1, 30, 1)
    
    # Architecture du modèle
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(30, 1), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Entraînement rapide sur les données du produit (10 epochs pour la démo)
    # En prod, on utiliserait un modèle pré-entraîné pour la vitesse
    model.fit(X_input, np.array([data_scaled[-1]]), epochs=15, verbose=0)
    
    # Prédiction
    pred_scaled = model.predict(X_input, verbose=0)
    vente_finale = scaler.inverse_transform(pred_scaled)
    
    return max(0, vente_finale[0][0])

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
