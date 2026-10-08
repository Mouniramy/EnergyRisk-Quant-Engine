import numpy as np
import pandas as pd

class GasStorageAsset:
    def __init__(self, max_capacity=1000.0, max_stockage=50.0, max_destockage=50.0, cout_stockage=0.05):
        self.max_capacity = max_capacity #Capacité totale du réservoir
        self.max_stockage = max_stockage #Volume max qu'on peut stocker par jour
        self.max_destockage = max_destockage #Volume max qu'on peut déstocker par jour
        self.cout_stockage = cout_stockage #Coût fixe d'injection par unité

    def optimize_storage_value(self, price_paths: np.ndarray):
        """
        Prend en entrée une matrice de trajectoires de prix simulées par Monte-Carlo 
        (jours x trajectoires) et optimise la valeur du stockage en prenant des décisions 
        simplifiées (achat bas / vente haut) sur chaque trajectoire.
        
        Retourne :
        - La valeur intrinsèque moyenne du stockage
        - Un résumé des gains générés par la flexibilité physique
        """
        n_days, n_paths = price_paths.shape
        total_payoffs = np.zeros(n_paths)

        # Simulation d'une stratégie de gestion du stockage par trajectoire
        for j in range(n_paths):
            path = price_paths[:, j]
            inventory = self.max_capacity * 0.5  # On commence avec un réservoir à moitié plein
            cash_flow = 0.0

            # Calcul d'une moyenne mobile simple pour décider d'acheter ou vendre
            # Si le prix est inférieur à la moyenne mobile, on injecte (achète). S'il est supérieur, on retire (vend).
            rolling_mean = pd.Series(path).rolling(window=5, min_periods=1).mean().values

            for t in range(n_days):
                current_price = path[t]
                m_mean = rolling_mean[t]

                if current_price < m_mean * 0.95 and inventory < self.max_capacity:
                    # Décision : Acheter / Injecter du gaz
                    volume = min(self.max_stockage, self.max_capacity - inventory)
                    cost = volume * (current_price + self.cout_stockage)
                    cash_flow -= cost
                    inventory += volume
                elif current_price > m_mean * 1.05 and inventory > 0:
                    # Décision : Vendre / Déstocker du gaz
                    volume = min(self.max_destockage, inventory)
                    revenue = volume * current_price
                    cash_flow += revenue
                    inventory -= volume

            # Valeur finale de la position (trésorerie + valeur résiduelle du gaz en stock au dernier prix)
            final_inventory_value = inventory * path[-1]
            total_payoffs[j] = cash_flow + final_inventory_value

        expected_value = np.mean(total_payoffs)
        value_std = np.std(total_payoffs)

        return {
            "expected_storage_value": round(expected_value, 2),
            "value_std": round(value_std, 2),
            "max_payoff": round(np.max(total_payoffs), 2),
            "min_payoff": round(np.min(total_payoffs), 2)
        }