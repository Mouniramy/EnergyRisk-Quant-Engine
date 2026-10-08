import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskEngine:
    def __init__(self, portfolio_values: pd.Series):
        self.portfolio_values = portfolio_values
        # Calcul des rendements/variations journalières en valeur monétaire ou en pourcentage
        self.pnl_series = portfolio_values.diff().dropna()

    def calculate_var_es(self, confidence_level=0.99):
        """
        Calcule la Value at Risk (VaR) et l'Expected Shortfall (ES) 
        par la méthode historique et paramétrique.
        """
        # Trier les PnL du pire au meilleur
        sorted_pnl = np.sort(self.pnl_series.values)
        
        # Indice correspondant au niveau de confiance
        index = int((1 - confidence_level) * len(sorted_pnl))
        
        # 1. VaR Historique
        var_historic = -sorted_pnl[index] if index < len(sorted_pnl) else -sorted_pnl[0]
        
        # 2. Expected Shortfall (ES) Historique : moyenne des pertes au-delà de la VaR
        tail_losses = sorted_pnl[:index]
        es_historic = -np.mean(tail_losses) if len(tail_losses) > 0 else var_historic

        # 3. VaR Paramétrique (en supposant une distribution normale)
        mean = np.mean(self.pnl_series)
        std = np.std(self.pnl_series)
        z_score = norm.ppf(confidence_level)
        var_parametric = -(mean - z_score * std)

        return {
            "var_historic": max(0.0, var_historic),
            "es_historic": max(0.0, es_historic),
            "var_parametric": max(0.0, var_parametric)
        }

    def run_stress_test(self, shock_factor_price=1.5, volatility_multiplier=2.0):
        """
        Simule un scénario de stress test énergétique extrême :
        - Un saut brutal des prix (ex: +50% sur un choc d'offre).
        - Une explosion de la volatilité.
        Retourne l'impact estimé sur la valeur finale du portefeuille.
        """
        last_val = self.portfolio_values.iloc[-1]
        historical_vol = np.std(self.pnl_series)
        stressed_loss = historical_vol * volatility_multiplier * shock_factor_price * np.sqrt(10) # choc sur 10 jours
        
        stressed_portfolio_value = last_val - abs(stressed_loss)
        total_drop_pct = (stressed_portfolio_value - last_val) / last_val * 100

        return {
            "initial_value": last_val,
            "stressed_value": max(0.0, stressed_portfolio_value),
            "estimated_loss": abs(stressed_loss),
            "drop_percentage": total_drop_pct
        }

    def run_backtest(self, confidence_level=0.99):
        window = 252  # Fenêtre d'un an
        if len(self.pnl_series) < window + 10:
            window = max(30, len(self.pnl_series) // 2)

        exceptions = 0
        total_tested = 0
        
        # Calcul glissant de la VaR
        for i in range(window, len(self.pnl_series)):
            history_slice = self.pnl_series.iloc[i-window:i]
            sorted_slice = np.sort(history_slice.values)
            idx = int((1 - confidence_level) * len(sorted_slice))
            var_t = -sorted_slice[idx] if idx < len(sorted_slice) else -sorted_slice[0]
            
            # Perte réelle observée au jour i 
            actual_pnl = self.pnl_series.iloc[i]
            
            if actual_pnl < -var_t:
                exceptions += 1
            total_tested += 1

        expected_exceptions = total_tested * (1 - confidence_level)
        
        return {
            "total_tested": total_tested,
            "exceptions": exceptions,
            "expected_exceptions": round(expected_exceptions, 1),
            "status": "VALIDÉ (Modèle robuste)" if exceptions <= expected_exceptions * 1.5 else "REJETÉ (Sous-estimation du risque)"
        }