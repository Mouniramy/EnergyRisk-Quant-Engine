# EnergyRisk Quant Engine

> Mini-logiciel de Risk Analytics et d'évaluation d'actifs pour les marchés de l'énergie (Gas/Power). J'ai développé cet outil pour démontrer des compétences en programmation Python, modélisation stochastique et gestion des risques.

---

## Aperçu du Projet

**EnergyRisk Quant Engine** est une application interactive construite avec **Streamlit** qui simule la dynamique des prix de l'énergie, calcule les métriques de risque de marché réglementaires, et évalue la flexibilité opérationnelle d'actifs de stockage physiques.

Ce projet s'inscrit dans une démarche d'ingénierie quantitative appliquée aux marchés du gaz naturel, combinant la finance de marché (gestion des risques de portefeuille) et l'industrie de l'énergie (optimisation d'infrastructures).

---

## Architecture & Fonctionnalités

L'application s'articule autour de trois briques principales :

### 1. Modèle Stochastique & Prix (Brique 1)
* **Calibration & Simulation :** Utilisation d'un modèle de retour à la moyenne (**Ornstein-Uhlenbeck**) calibré sur un historique de 2 ans de données de marché (`yfinance`) pour capturer la dynamique propre aux matières premières énergétiques.
* **Trajectoires de Monte-Carlo :** Génération de scénarios futurs multi-periodes (`n_paths` trajectoires sur `n_days` jours)[cite: 2].
* **Visualisation interactive :** Graphique Plotly combinant l'historique récent (fenêtre glissante optimisée pour la lisibilité) et un éventail de trajectoires futures simulées[cite: 2].

### 2. Risque de Marché & Stress Testing (Brique 2)
* **Value at Risk (VaR) Historique :** Calcul de la VaR au seuil de 95% ou 99% sur un portefeuille synthétique de référence[cite: 2].
* **Expected Shortfall (ES) :** Évaluation de la perte moyenne conditionnelle au-delà du seuil de VaR pour capturer le risque de queue de distribution.
* **Backtesting :** Vérification de la robustesse du modèle de risque par analyse des exceptions historiques.
* **Module de Stress Test :** Application d'un choc combiné de marché (explosion des prix et volatilité extrême) pour simuler l'impact sur la valeur du portefeuille et identifier les seuils de rupture[cite: 2].

### 3. Valorisation du Stockage Physique (Brique 3)
* **Flexibilité Opérationnelle :** Modélisation d'un réservoir de stockage souterrain soumis à des contraintes physiques strictes (capacité maximale `max_capacity`, taux d'injection et de retrait `max_injection_rate`)[cite: 2].
* **Optimisation par Scénarios :** Résolution de la stratégie optimale d'arbitrage (acheter bas en été, déstocker en hiver) à travers l'ensemble des trajectoires de Monte-Carlo[cite: 2].
* **Métriques clés :** Restitution de la valeur intrinsèque attendue, du gain maximal potentiel et de l'écart-type des profits[cite: 2].

---

## Structure du Codebase

```text
EnergyRisk-QuantEngine/
│
├── app.py                  # Interface utilisateur principale (Streamlit)
├── requirements.txt        # Dépendances du projet
└── src/
    ├── stochastic.py       # Moteur stochastique (Ornstein-Uhlenbeck & Monte-Carlo)
    ├── risk_engine.py      # Moteur de risque (VaR, ES, Backtesting & Stress Test)
    └── physical_asset.py   # Modélisation et optimisation de l'actif de stockage physique