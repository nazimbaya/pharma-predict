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
# 3. L'APPLICATION UTILISATEUR (Version Pro)
# ==========================================
st.info("💡 Déposez votre export de stock (Excel ou CSV)")
uploaded_file = st.file_uploader("Fichier d'historique", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    df = pd.read_csv(uploaded_file) 
    st.success("Fichier chargé avec succès !")
    st.write("---")
    
    # --- DÉBUT DE LA ZONE DE MAPPING ---
    st.write("### ⚙️ Étape 1 : Identifiez vos colonnes")
    st.caption("Aidez l'IA à comprendre votre fichier en associant vos colonnes.")
    
    colonnes_disponibles = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        col_nom = st.selectbox("1. Colonne du Nom du Produit :", colonnes_disponibles)
    with col2:
        col_stock = st.selectbox("2. Colonne du Stock Actuel :", colonnes_disponibles)
        
    cols_historique = st.multiselect("3. Colonnes de l'historique des ventes (ex: les 30 derniers jours) :", colonnes_disponibles)
    # --- FIN DE LA ZONE DE MAPPING ---

    st.write("---")
    
    # On ne permet de lancer l'IA que si l'utilisateur a choisi ses colonnes d'historique
    if len(cols_historique) < 2:
        st.warning("⚠️ Veuillez sélectionner au moins 2 colonnes d'historique pour que l'IA puisse détecter une tendance.")
    else:
        if st.button("🚀 Étape 2 : Lancer le Diagnostic IA"):
            st.write("### 🚨 Analyse Prédictive des Stocks")
            
            # Affichage de l'en-tête du tableau des résultats
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write("**Produit**")
            c2.write("**Prédiction Demain**")
            c3.write("**Statut**")
            st.write("---")
            
            # Boucle d'analyse sur les colonnes choisies par le pharmacien !
            for index, row in df.iterrows():
                nom = row[col_nom]          # Utilise le choix de l'utilisateur
                stock = row[col_stock]      # Utilise le choix de l'utilisateur
                
                # Récupère uniquement les colonnes d'historique sélectionnées
                historique = row[cols_historique].values.astype(float) 
                
                # Appel du Machine Learning (ta fonction Scikit-Learn)
                prediction = predire_prochaine_vente(historique)
                
                # Calcul de sécurité
                jours_restants = stock / prediction if prediction > 0.1 else 99
                
                # Affichage des alertes
                if jours_restants < 5:
                    st.error(f"**{nom}** | Reste: {int(jours_restants)} jours")
                    st.caption(f"Vente estimée demain : {prediction:.1f} boîtes")
                elif jours_restants < 15:
                    st.warning(f"**{nom}** | Reste: {int(jours_restants)} jours")
