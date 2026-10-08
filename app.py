import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.stochastic import EnergyStochasticEngine
from src.risk_engine import RiskEngine
from src.physical_asset import GasStorageAsset

# Configuration de la page Streamlit
st.set_page_config(
    page_title="EnergyRisk Quant Engine",
    page_icon="o",
    layout="wide"
)

st.title("EnergyRisk Quant Engine")
st.markdown("""
Mini-logiciel de Risk Analytics et d'évaluation d'actifs pour les marchés de l'énergie (Natural Gas). 
J'ai développé cet outil pour démontrer des compétences en programmation Python, modélisation stochastique et gestion des risques.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Paramètres de Simulation")
ticker = st.sidebar.text_input("Ticker Marché (Yahoo Finance)", value="NG=F")
n_paths = st.sidebar.slider("Trajectoires Monte-Carlo", min_value=100, max_value=2000, value=500, step=100)
n_days = st.sidebar.slider("Horizon de simulation (jours)", min_value=10, max_value=90, value=30, step=5)
confidence_level = st.sidebar.selectbox("Niveau de Confiance VaR / ES", [0.95, 0.99], index=1)

st.sidebar.markdown("---")
st.sidebar.header("Paramètres Stockage de Gaz")
max_capacity = st.sidebar.number_input("Capacité Max Stockage (MWh)", value=1000.0, step=100.0)
max_injection = st.sidebar.number_input("Taux Max Injection / jour", value=50.0, step=10.0)

# --- INITIALISATION DES DONNÉES ET DU MOTEUR ---
@st.cache_resource
def load_engine(ticker_symbol):
    return EnergyStochasticEngine(ticker=ticker_symbol)

engine = load_engine(ticker)
historical_prices = engine.historical_prices

# Lancer les simulations Monte-Carlo
simulated_paths = engine.simulate_monte_carlo(n_paths=n_paths, n_days=n_days)

# Créer un portefeuille synthétique basé sur l'historique pour le risk engine
portfolio_synth = historical_prices * 10000 

# --- ORGANISATION EN ONGLETS ---
tab1, tab2, tab3 = st.tabs(["1. Modèle Stochastique & Prix", "2. Risque de Marché & Stress Test", "3. Valorisation Stockage Physique"])

with tab1:
    st.subheader("Modélisation Stochastique (Ornstein-Uhlenbeck & Monte-Carlo)")
    st.markdown("Comparaison entre l'historique réel des prix de l'énergie et les trajectoires futures simulées.")

    fig = go.Figure()
    # Courbe historique
    fig.add_trace(go.Scatter(
        x=historical_prices.index[-100:], 
        y=historical_prices.values[-100:], 
        mode='lines', 
        name='Historique Réel', 
        line=dict(color='black', width=2)
    ))
    
    # Trajectoires Monte-Carlo
    future_dates = pd.date_range(start=historical_prices.index[-1], periods=n_days+1, freq='B')
    for i in range(min(50, n_paths)): # Affichage de 50 chemins pour la lisibilité
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=simulated_paths[:, i], 
            mode='lines', 
            line=dict(color='rgba(0, 150, 255, 0.15)', width=1),
            showlegend=False
        ))

    fig.update_layout(
        title=f"Simulation de Prix - {ticker}",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Analyse du Risque de Marché, VaR & Stress Testing")
    
    risk_eng = RiskEngine(portfolio_synth)
    metrics = risk_eng.calculate_var_es(confidence_level=confidence_level)
    backtest = risk_eng.run_backtest(confidence_level=confidence_level)
    stress = risk_eng.run_stress_test(shock_factor_price=1.8, volatility_multiplier=2.5)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"VaR ({int(confidence_level*100)}%) Historique", f"{metrics['var_historic']:,.2f} $")
    col2.metric(f"Expected Shortfall (ES)", f"{metrics['es_historic']:,.2f} $")
    col3.metric("Statut Backtesting", backtest['status'])

    st.markdown("---")
    st.markdown("### Module de Stress Test Énergétique")
    st.markdown("Simulation d'un choc combiné de marché : explosion des prix et volatilité extrême.")
    
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("Valeur Initiale", f"{stress['initial_value']:,.2f} $")
    s_col2.metric("Valeur sous Stress", f"{stress['stressed_value']:,.2f} $", delta=f"{stress['drop_percentage']:.2f}%", delta_color="inverse")
    s_col3.metric("Perte Estimée", f"{stress['estimated_loss']:,.2f} $")

with tab3:
    st.subheader("Optimisation & Valorisation d'un Actif Physique (Stockage de Gaz)")
    st.markdown("Évaluation de la flexibilité opérationnelle d'un réservoir de stockage soumis aux trajectoires de prix.")

    storage = GasStorageAsset(
        max_capacity=max_capacity,
        max_injection_rate=max_injection,
        max_withdrawal_rate=max_injection
    )

    with st.spinner("Calcul de l'optimisation des flux de stockage par Monte-Carlo..."):
        storage_results = storage.optimize_storage_value(simulated_paths)

    st_col1, st_col2, st_col3 = st.columns(3)
    st_col1.metric("Valeur Intrinsèque Attendue", f"{storage_results['expected_storage_value']:,.2f} $")
    st_col2.metric("Gain Max Potentiel", f"{storage_results['max_payoff']:,.2f} $")
    st_col3.metric("Écart-type des Payoffs", f"{storage_results['value_std']:,.2f} $")

    st.info("**Principe métier :** Ce chiffre représente la valeur théorique de la flexibilité (acheter quand le gaz est bas, déstocker quand les prix explosent).")