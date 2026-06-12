import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier

# MASQUE DE SÉCURITÉ GLOBAL (USER-AGENT)
import requests
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
session.headers.update(headers)

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Dashboard Quantitative CAC 40", layout="wide", page_icon="📈")

st.title("📊 Application Finale d'Optimisation de Portefeuille (Composants du CAC 40)")
st.write("Ce tableau de bord extrait automatiquement la liste des composants du CAC 40 en direct et applique les modèles d'allocation de Markowitz.")

# --- FONCTION AUTOMATISÉE ET SÉCURISÉE POUR LE CAC 40 ---
@st.cache_data
def recuperer_actifs_cac40():
    try:
        url = "https://fr.wikipedia.org/wiki/CAC_40"
        # Ajout des headers pour éviter l'erreur HTTP 403 Forbidden
        req = requests.get(url, headers=headers)
        tables = pd.read_html(req.text)
        
        for table in tables:
            if 'Ticker' in table.columns:
                table['Ticker_YF'] = table['Ticker'].apply(lambda x: str(x).strip() if str(x).endswith('.PA') else f"{str(x).strip()}.PA")
                return dict(zip(table['Société'], table['Ticker_YF']))
    except Exception as e:
        pass
    
    # Dictionnaire de secours complet (Fallback) au cas où Wikipédia est inaccessible
    return {
        "Airbus": "AIR.PA", "TotalEnergies": "TTE.PA", "LVMH": "MC.PA", "BNP Paribas": "BNP.PA",
        "Sanofi": "SAN.PA", "L'Oreal": "OR.PA", "AXA": "CS.PA", "Schneider Electric": "SU.PA",
        "Air Liquide": "AI.PA", "Danone": "BN.PA", "Vinci": "DG.PA", "Pernod Ricard": "RI.PA"
    }

# Charger les actifs
actifs_cac40 = recuperer_actifs_cac40()

# --- BARRE LATÉRALE ---
st.sidebar.header("⚙️ Configuration du Portefeuille")

noms_selectionnes = st.sidebar.multiselect(
    "Sélectionnez les actions à inclure :",
    options=list(actifs_cac40.keys()),
    default=list(actifs_cac40.keys())[:6]
)

if st.sidebar.button("✨ Sélectionner tout le CAC 40"):
    noms_selectionnes = list(actifs_cac40.keys())
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🌍 Comparaison Marché Réel")
indices_marche = {
    "CAC 40 (Indice global)": "^FCHI",
    "S&P 500 (USA)": "^SPX",
    "Euro Stoxx 50 (Europe)": "^STOXX50E"
}
nom_indice_selectionne = st.sidebar.selectbox("Choisir l'indice de référence :", list(indices_marche.keys()))
ticker_indice = indices_marche[nom_indice_selectionne]

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Date de début :", pd.to_datetime("2010-01-01"), min_value=pd.to_datetime("1990-01-01"), max_value=pd.to_datetime("2050-12-31"))
end_date = st.sidebar.date_input("Date de fin :", pd.to_datetime("2025-12-31"), min_value=pd.to_datetime("1990-01-01"), max_value=pd.to_datetime("2050-12-31"))

capital_initial = st.sidebar.number_input("Capital initial (€) :", min_value=1000, value=10000, step=1000)
taux_sans_risque = st.sidebar.slider("Taux sans risque (%) :", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100

# --- TRAITEMENT FINANCIER ---
if len(noms_selectionnes) < 2:
    st.warning("⚠️ Veuillez sélectionner au moins 2 actions.")
else:
    with st.spinner(f"🌐 Extraction des données Yahoo Finance..."):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Téléchargement des actions
        dict_prix = {}
        for nom in noms_selectionnes:
            ticker = actifs_cac40[nom]
            try:
                df_single = yf.download(ticker, start=start_str, end=end_str, auto_adjust=True, session=session, progress=False)
                if not df_single.empty:
                    # Extraction propre de la série de clôture
                    if isinstance(df_single.columns, pd.MultiIndex):
                        dict_prix[nom] = df_single.iloc[:, 0].squeeze()
                    else:
                        dict_prix[nom] = df_single['Close'].squeeze()
            except:
                pass
        
        # Téléchargement de l'indice
        df_indice = pd.DataFrame()
        try:
            df_single_indice = yf.download(ticker_indice, start=start_str, end=end_str, auto_adjust=True, session=session, progress=False)
            if not df_single_indice.empty:
                # 🛠️ Correction du bug .to_frame()
                if isinstance(df_single_indice.columns, pd.MultiIndex):
                    df_indice['Indice'] = df_single_indice.iloc[:, 0]
                else:
                    df_indice['Indice'] = df_single_indice['Close']
        except Exception as e:
            st.warning(f"Erreur d'indice : {e}")

        if dict_prix:
            df_prix = pd.DataFrame(dict_prix).ffill().bfill()
        else:
            df_prix = pd.DataFrame()

    if df_prix.empty or len(df_prix) < 5 or df_indice.empty:
        st.error("❌ Les données n'ont pas pu être chargées. Modifiez la plage de dates.")
        st.stop()

    # --- MODELE DE MARKOWITZ ---
    mu = expected_returns.mean_historical_return(df_prix)
    S = risk_models.CovarianceShrinkage(df_prix).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    
    try:
        poids_bruts = ef.max_sharpe(risk_free_rate=taux_sans_risque)
        poids_optimises = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=taux_sans_risque)
        
        # --- KPI ---
        st.subheader("🏆 Profil Théorique du Portefeuille Optimal (Max Sharpe)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement Annuel Attendu", f"{ret:.2%}")
        col2.metric("Volatilité Annuelle (Risque)", f"{vol:.2%}")
        col3.metric("Ratio de Sharpe", f"{sharpe:.2f}")

        st.markdown("---")

        # --- GRAPH ET POIDS ---
        col_gauche, col_droite = st.columns(2)
        with col_gauche:
            st.subheader("⚖️ Répartition de l'allocation")
            df_poids = pd.DataFrame(list(poids_optimises.items()), columns=["Entreprise", "Allocation (%)"])
            df_poids["Allocation (%)"] = df_poids["Allocation (%)"] * 100
            df_poids_actifs = df_poids[df_poids["Allocation (%)"] > 0].sort_values(by="Allocation (%)", ascending=False)
            st.dataframe(df_poids_actifs.style.format({"Allocation (%)": "{:.2f} %"}), use_container_width=True)

        with col_droite:
            st.subheader("🍕 Ventilation graphique (Poids > 0%)")
            poids_filtres = {k: v for k, v in poids_optimises.items() if v > 0.005}
            if poids_filtres:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                ax_pie.pie(poids_filtres.values(), labels=poids_filtres.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
                ax_pie.axis('equal')
                st.pyplot(fig_pie)

        st.markdown("---")

        # --- BACKTESTING NETTOYÉ CONTRE LES ANOMALIES DE MARCHÉ ---
        st.subheader(f"📈 Suivi Historique du Capital vs Indice de Référence ({nom_indice_selectionne})")
        
        # 🛠️ Nettoyage financier : Calcul des rendements via PyPortfolioOpt pour éliminer les aberrations
        rendements_quotidiens = expected_returns.returns_from_prices(df_prix).dropna()
        
        # Élimination radicale des lignes aberrantes (ex: variations > 50% en une journée dues à un bug de split)
        rendements_quotidiens = rendements_quotidiens[(rendements_quotidiens.abs() < 0.5).all(axis=1)]
        
        poids_ordonnes = np.array([poids_optimises[nom] for nom in df_prix.columns])
        poids_benchmark = np.array([1/len(df_prix.columns)] * len(df_prix.columns))

        rendement_portefeuille = rendements_quotidiens.dot(poids_ordonnes)
        rendement_benchmark = rendements_quotidiens.dot(poids_benchmark)

        # Alignement de l'indice de marché
        df_indice_aligne = df_indice.reindex(rendements_quotidiens.index).ffill().bfill()
        rendement_indice = expected_returns.returns_from_prices(df_indice_aligne).dropna()

        communes_idx = rendement_portefeuille.index.intersection(rendement_indice.index)
        
        # Calcul des trajectoires cumulées propres
        valeur_portefeuille = capital_initial * (1 + rendement_portefeuille.loc[communes_idx]).cumprod()
        valeur_benchmark = capital_initial * (1 + rendement_benchmark.loc[communes_idx]).cumprod()
        valeur_marche = capital_initial * (1 + rendement_indice.loc[communes_idx]["Indice"]).cumprod()

        # Graphique
        fig_backtest, ax_backtest = plt.subplots(figsize=(12, 5))
        ax_backtest.plot(valeur_portefeuille.index, valeur_portefeuille.values, label='Allocation Algorithmique (Max Sharpe)', color='darkgreen', lw=2.5)
        ax_backtest.plot(valeur_benchmark.index, valeur_benchmark.values, label='Gestion Naïve (1/N Équiréparti)', color='gray', linestyle='--', alpha=0.6)
        ax_backtest.plot(valeur_marche.index, valeur_marche.values, label=f'Marché Réel ({nom_indice_selectionne})', color='royalblue', lw=2)
        ax_backtest.set_ylabel('Valeur patrimoniale (€)')
        ax_backtest.grid(True, alpha=0.3)
        ax_backtest.legend()
        st.pyplot(fig_backtest)

        # Bilans terminaux
        v_finale_opt = valeur_portefeuille.iloc[-1]
        v_finale_bench = valeur_benchmark.iloc[-1]
        v_finale_marche = valeur_marche.iloc[-1]
        
        perf_opt = (v_finale_opt / capital_initial - 1) * 100
        perf_bench = (v_finale_bench / capital_initial - 1) * 100
        perf_marche = (v_finale_marche / capital_initial - 1) * 100

        st.markdown("### 📊 Résultats financiers finaux :")
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.success(f"🚀 Portefeuille Optimisé  \n**{v_finale_opt:,.2f} €** ({perf_opt:+.2f}%)")
        col_b2.info(f"⚖️ Gestion Naïve 1/N  \n**{v_finale_bench:,.2f} €** ({perf_bench:+.2f}%)")
        col_b3.metric(f"🏢 {nom_indice_selectionne}", f"{v_finale_marche:,.2f} €", f"{perf_marche:+.2f}%")

    except Exception as e:
        st.error(f"Erreur d'optimisation : {e}")