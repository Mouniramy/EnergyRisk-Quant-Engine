import numpy as np
import pandas as pd
import yfinance as yf

class EnergyStochasticEngine:
    def __init__(self, ticker="NG=F", periode="2y"):
        #Initialise le moteur avec un ticker par défaut (Natural Gas Futures) et une période de récupération des données historiques.
        self.ticker = ticker
        self.periode = periode
        self.historical_prices = self.fetch_data()

    def fetch_data(self) -> pd.Series:
        try:
            print(f"Téléchargement des données pour {self.ticker}...")
            data = yf.download(self.ticker, periode=self.periode, progress=False)
            
            # Gestion de la structure des colonnes de yfinance selon les versions
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'].iloc[:, 0]
            else:
                prices = data['Close']
                
            prices = prices.dropna()
            
            if len(prices) < 50:
                raise ValueError("Pas assez de données récupérées.")
            
            print(f"Données téléchargées avec succès : {len(prices)} points de données.")
            return prices
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement ({e}). Génération de données synthétiques de secours...")

    def calibrate_ou_parameters(self):
        """
        Calibre les paramètres du modèle Ornstein-Uhlenbeck (Mean-Reversion) 
        à partir des données historiques réelles :
        dX_t = kappa * (theta - X_t) * dt + sigma * dW_t
        """
        prices = self.historical_prices.values
        dt = 1.0 / 252.0  # Pas de temps journalier
        
        # Régression linéaire pour estimer la vitesse de retour à la moyenne (kappa) et la moyenne long terme (theta)
        x_t = prices[:-1]
        x_t1 = prices[1:]
        
        slope, intercept = np.polyfit(x_t, x_t1, 1)
        
        kappa = -np.log(slope) / dt
        theta = intercept / (1 - slope)
        
        # Volatilité des résidus
        residuals = x_t1 - (slope * x_t + intercept)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        return kappa, theta, sigma

    def simulate_monte_carlo(self, n_paths=1000, n_days=30):
        kappa, theta, sigma = self.calibrate_ou_parameters()
        dt = 1.0 / 252.0
        
        last_price = self.historical_prices.iloc[-1]
        
        # Matrice pour stocker les simulations (jours x trajectoires)
        simulated_paths = np.zeros((n_days + 1, n_paths))
        simulated_paths[0, :] = last_price
        
        sqrt_dt = np.sqrt(dt)
        
        for t in range(1, n_days + 1):
            # Incrément stochastique (loi normale)
            dW = np.random.normal(0, sqrt_dt, n_paths)
            # Équation de l'Ornstein-Uhlenbeck discrétisée (Euler-Maruyama)
            dr = kappa * (theta - simulated_paths[t-1, :]) * dt + sigma * dW
            simulated_paths[t, :] = simulated_paths[t-1, :] + dr
            
        # S'assurer qu'aucun prix n'est négatif
        simulated_paths = np.maximum(simulated_paths, 0.01)
        
        return simulated_paths