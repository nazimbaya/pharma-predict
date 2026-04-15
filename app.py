import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. CONFIGURATION DE L'INTERFACE
# ==========================================
st.set_page_config(page_title="Pharma-Predict | Pharmakode", page_icon="💊")

st.title("💊 Pharma-Predict (by Pharmakode)")
st.subheader("Anticipez vos ruptures de stock grâce au Machine Learning")
st.write("---")

# ==========================================
# 2. LE CERVEAU IA (Scikit-Learn)
# ==========================================
def predire_prochaine_vente(historique_30j):
    # Transformation des 30 jours pour l'algorithme
    X = np.array(range(len(historique_30j))).reshape(-1, 1)
    y = np.array(historique_30j)
    
    # La Régression Polynomiale (détecte les accélérations de ventes)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédit le jour 31
    X_31 = poly.transform([[31]])
    prediction = model.predict(X_31)[0]
    
    return max(0, prediction)

# ==========================================
# 3. L'APPLICATION UTILISATEUR
# ==========================================
st.info("💡 Déposez le fichier de test de 30 jours (test_pharmacie_30j.csv)")
uploaded_file = st.file_uploader("Fichier d'historique (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Fichier chargé avec succès !")
    
    # On identifie automatiquement les colonnes de jours (Jour_1, Jour_2, etc.)
    colonnes_jours = [col for col in df.columns if 'Jour_' in col]
    
    if st.button("🚀 Lancer le Diagnostic IA"):
        st.write("### 🚨 Analyse Prédictive des Stocks")
        
        # On crée des colonnes pour un affichage propre
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write("**Produit**")
        col2.write("**Prédiction Demain**")
        col3.write("**Statut**")
        st.write("---")
        
        # On analyse chaque médicament
        for index, row in df.iterrows():
            nom = row['DESIGNATION']
            stock = row['STOCK_ACTUEL']
            historique = row[colonnes_jours].values.astype(float)
            
            # Appel du Machine Learning
            prediction = predire_prochaine_vente(historique)
            
            # Calcul de sécurité
            jours_restants = stock / prediction if prediction > 0.1 else 99
            
            # Affichage visuel selon l'urgence
            if jours_restants < 5:
                st.error(f"**{nom}** | Reste: {int(jours_restants)} jours")
                st.caption(f"Vente estimée demain : {prediction:.1f} boîtes")
            elif jours_restants < 15:
                st.warning(f"**{nom}** | Reste: {int(jours_restants)} jours")
            else:
                st.success(f"**{nom}** | Stock sain (> 15 j)")

st.write("---")
st.caption("Une solution propulsée par Pharmakode - Technologie d'aide à la décision pour l'officine.")
