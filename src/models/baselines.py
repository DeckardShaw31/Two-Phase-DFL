"""
Two-Stage Predict-then-Optimize (PTO) Baselines for Perishable Inventory-Routing.

Comprehensive 10-Baseline Suite for Multi-Period Perishable Inventory-Routing:
  1. RidgePTO (Ridge L2 Regularized Linear Regression + MILP)
  2. ElasticNetPTO (L1+L2 ElasticNet Regression + MILP)
  3. RandomForestPTO (Bagging Ensemble Trees + MILP)
  4. GradientBoostingPTO (Boosting Trees + MILP)
  5. LightGBMPTO (Symmetric MSE LightGBM GBDT + MILP)
  6. LightGBMQuantilePTO (Asymmetric Quantile / Critical Fractile GBDT + MILP)
  7. SafetyStockPTO (Ridge Point Forecast + Empirical Safety Buffer + MILP)
  8. MLPPTO (Deep Neural Network trained on MSE + MILP)
  9. KNNSAAModel (K-Nearest Neighbors Sample Average Approximation + MILP)
  10. TemporalSeqNetPTO (Deep Recurrent LSTM/GRU Sequence Net with Store Embeddings + MILP)
"""

import copy

import lightgbm as lgb
import numpy as np
import torch
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.optimization.pirp_model import PIRPSolution, solve_pirp


class DemandPredictorMLP(nn.Module):
    """Multi-Layer Perceptron for demand forecasting."""

    def __init__(self, in_features: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TemporalSequenceModule(nn.Module):
    """
    PyTorch Recurrent Sequence Module (LSTM/GRU) processing features across horizon_T.
    Includes learnable store entity embeddings and Softplus non-negative output head.
    """

    def __init__(
        self,
        in_features: int = 14,
        num_stores: int = 10,
        horizon_T: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        store_embed_dim: int = 8,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_stores = num_stores
        self.horizon_T = horizon_T
        self.store_embed_dim = store_embed_dim
        self.bidirectional = bidirectional

        if store_embed_dim > 0:
            self.store_embed = nn.Embedding(num_stores, store_embed_dim)
            rnn_in_dim = in_features + store_embed_dim
        else:
            self.store_embed = None
            rnn_in_dim = in_features

        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def calibrate_bias(self, target_mean: float):
        """Calibrates output layer bias to the target mean demand."""
        bias_val = float(np.log(np.exp(min(50.0, target_mean)) - 1.0)) if target_mean < 20.0 else float(target_mean)
        self.head[-2].bias.data.fill_(bias_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling 2D (M, F), 3D (N, T, F), and 4D (B, N, T, F) or (B, T, N, F).
        """
        if x.ndim == 2:
            num_samples, _ = x.shape
            total_per_inst = self.num_stores * self.horizon_T
            if num_samples % total_per_inst == 0 and num_samples >= total_per_inst:
                # Can be structured into (B, num_stores, horizon_T, F)
                B = num_samples // total_per_inst
                x_4d = x.view(B, self.num_stores, self.horizon_T, self.in_features)
                out_4d = self.forward(x_4d)  # (B, N, T)
                return out_4d.view(num_samples)

            # Fallback for arbitrary sample count: assign cyclic store IDs
            store_ids = torch.arange(num_samples, device=x.device) % self.num_stores
            if self.store_embed is not None:
                embeds = self.store_embed(store_ids).unsqueeze(1)  # (M, 1, d_e)
                x_cat = torch.cat([x.unsqueeze(1), embeds], dim=-1)  # (M, 1, F + d_e)
            else:
                x_cat = x.unsqueeze(1)
            rnn_out, _ = self.rnn(x_cat)
            out = self.head(rnn_out).squeeze(-1).squeeze(-1)  # (M,)
            return out

        if x.ndim == 3:
            # Single instance (N, T, F)
            x_4d = x.unsqueeze(0)  # (1, N, T, F)
            out_4d = self.forward(x_4d)  # (1, N, T)
            return out_4d.squeeze(0)  # (N, T)

        if x.ndim == 4:
            B, s1, s2, _ = x.shape
            if s1 == self.horizon_T and s2 == self.num_stores:
                x_perm = x.permute(0, 2, 1, 3)  # -> (B, N, T, F)
            else:
                x_perm = x  # (B, N, T, F)
            N, T = self.num_stores, self.horizon_T

            if self.store_embed is not None:
                store_ids = torch.arange(N, device=x.device)
                embeds = self.store_embed(store_ids)  # (N, store_embed_dim)
                embeds = embeds.unsqueeze(0).unsqueeze(2).expand(B, N, T, self.store_embed_dim)
                x_comb = torch.cat([x_perm, embeds], dim=-1)  # (B, N, T, F + d_e)
            else:
                x_comb = x_perm

            x_seq = x_comb.contiguous().view(B * N, T, -1)
            rnn_out, _ = self.rnn(x_seq)
            out = self.head(rnn_out).squeeze(-1)  # (B * N, T)
            return out.view(B, N, T)

        raise ValueError(f"Unsupported tensor shape: {x.shape}")


class BasePTOModel:
    """
    Abstract base class for Two-Stage PTO models.
    Provides automated multidimensional reshaping between 2D tabular features (M, F)
    and higher-order spatiotemporal tensors (B, N, T, F), (B, T, N, F), or (N, T, F).
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        raise NotImplementedError

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the underlying model.
        Accepts:
          - 2D tabular array: (M, F) and (M,)
          - 3D/4D tensor: (B, N, T, F) / (B, T, N, F) and (B, N, T) / (B, T, N)
        """
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if X_arr.ndim > 2:
            X_flat = X_arr.reshape(-1, X_arr.shape[-1])
            y_flat = y_arr.reshape(-1)
        else:
            X_flat = X_arr
            y_flat = y_arr.reshape(-1)
        self._fit_flat(X_flat, y_flat)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts non-negative demand, preserving input shape:
          - (M, F) -> returns (M,)
          - (N, T, F) -> returns (N, T)
          - (B, N, T, F) -> returns (B, N, T)
          - (B, T, N, F) -> returns (B, T, N)
        """
        X_arr = np.asarray(X, dtype=np.float32)
        orig_shape = X_arr.shape
        if X_arr.ndim > 2:
            X_flat = X_arr.reshape(-1, orig_shape[-1])
            preds_flat = self._predict_flat(X_flat)
            return np.maximum(0.0, preds_flat).reshape(orig_shape[:-1])
        else:
            preds = self._predict_flat(X_arr)
            return np.maximum(0.0, preds)

    def optimize(
        self,
        predicted_demand: np.ndarray,
        distance_matrix: np.ndarray | None = None,
        continuous_relaxation: bool = False,
        shelf_life_tau: int = 3,
        holding_cost_h: float = 1.0,
        aging_surcharge_eps: float = 0.01,
        stockout_penalty_pu: float = 20.0,
        spoilage_penalty_ps: float = 5.0,
        vehicle_capacity_Q: float = 100.0,
        store_capacity_Ci: float = 70.0,
        time_limit_sec: float | None = 3.0,
    ) -> PIRPSolution:
        """Solves downstream PIRP optimization using predicted demand."""
        return solve_pirp(
            demand=predicted_demand,
            distance_matrix=distance_matrix,
            continuous_relaxation=continuous_relaxation,
            shelf_life_tau=shelf_life_tau,
            holding_cost_h=holding_cost_h,
            aging_surcharge_eps=aging_surcharge_eps,
            stockout_penalty_pu=stockout_penalty_pu,
            spoilage_penalty_ps=spoilage_penalty_ps,
            vehicle_capacity_Q=vehicle_capacity_Q,
            store_capacity_Ci=store_capacity_Ci,
            time_limit_sec=time_limit_sec,
        )


class RidgePTO(BasePTOModel):
    """Two-Stage PTO baseline 1: Ridge Regression (L2 penalty) + MILP."""

    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge_PTO")
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)


class ElasticNetPTO(BasePTOModel):
    """
    Two-Stage PTO baseline 2: ElasticNet Regression (L1 + L2 penalty) + MILP.
    Balances sparsity and grouping for collinear demand lag features.
    """

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        super().__init__("ElasticNet_PTO")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
            max_iter=2000,
        )

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)


ElasticNetBaseline = ElasticNetPTO


class RandomForestPTO(BasePTOModel):
    """
    Two-Stage PTO baseline 3: Random Forest Regressor (Bagging Ensemble) + MILP.
    Models non-linear feature interactions without sequential boosting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        random_state: int = 42,
    ):
        super().__init__("RandomForest_PTO")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=1,
        )

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        self.model.fit(X_flat, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)


RandomForestBaseline = RandomForestPTO


class GradientBoostingPTO(BasePTOModel):
    """
    Two-Stage PTO baseline 4: Gradient Tree Boosting Regressor (GBDT) + MILP.
    Sequential tree boosting minimizing squared error loss.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        super().__init__("GradientBoosting_PTO")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        self.model.fit(X_flat, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)


GradientBoostingBaseline = GradientBoostingPTO


class LightGBMPTO(BasePTOModel):
    """Two-Stage PTO baseline 5: LightGBM GBDT (Symmetric MSE) + MILP."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.05):
        super().__init__("LightGBM_PTO")
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        self.model.fit(X_flat, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)


class LightGBMQuantilePTO(BasePTOModel):
    """
    Two-Stage PTO baseline 6: LightGBM Asymmetric Quantile Regressor + MILP.
    Trained with pinball loss at the Newsvendor critical fractile:
        (1) Myopic single-period: alpha_myopic = p_u / (p_u + p_s)
        (2) Multi-period tau-amortized: beta_tau* = p_u / (p_u + h + p_s / tau)
    """

    def __init__(
        self,
        stockout_penalty_pu: float = 20.0,
        spoilage_penalty_ps: float = 5.0,
        holding_cost_h: float = 1.0,
        shelf_life_tau: int = 3,
        critical_fractile: float | None = None,
        fractile_type: str = "tau_amortized",
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
    ):
        super().__init__("LightGBM_Quantile_PTO")
        self.fractile_type = fractile_type
        if critical_fractile is not None:
            self.critical_fractile = float(critical_fractile)
        elif fractile_type == "myopic":
            self.critical_fractile = float(stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps))
        elif fractile_type in ("tau_amortized", "amortized"):
            if shelf_life_tau is not None and shelf_life_tau > 0:
                self.critical_fractile = float(
                    stockout_penalty_pu / (stockout_penalty_pu + holding_cost_h + spoilage_penalty_ps / shelf_life_tau)
                )
            else:
                self.critical_fractile = float(
                    stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps + holding_cost_h)
                )
        elif shelf_life_tau is not None and shelf_life_tau > 0:
            self.critical_fractile = float(
                stockout_penalty_pu / (stockout_penalty_pu + holding_cost_h + spoilage_penalty_ps / shelf_life_tau)
            )
        else:
            self.critical_fractile = float(
                stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps + holding_cost_h)
            )

        self.model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=self.critical_fractile,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        self.model.fit(X_flat, y_flat)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)


LightGBMQuantileBaseline = LightGBMQuantilePTO


class SafetyStockPTO(BasePTOModel):
    """
    Two-Stage PTO baseline 7: Safety-Stock Heuristic + MILP.
    Industry standard rule-of-thumb: Ridge point forecast + static safety buffer:
        d_safe = d_point + z * sigma
    where z = norm.ppf(critical_fractile) and sigma is empirical training residual standard deviation.
    Supports fractile_type="myopic" (alpha = 0.80) and "tau_amortized" (beta = 0.8824).
    """

    def __init__(
        self,
        alpha_ridge: float = 1.0,
        stockout_penalty_pu: float = 20.0,
        spoilage_penalty_ps: float = 5.0,
        holding_cost_h: float = 1.0,
        shelf_life_tau: int = 3,
        critical_fractile: float | None = None,
        fractile_type: str = "tau_amortized",
        z_factor: float | None = None,
    ):
        super().__init__("SafetyStock_PTO")
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha_ridge)
        self.fractile_type = fractile_type
        if critical_fractile is not None:
            self.critical_fractile = float(critical_fractile)
        elif fractile_type == "myopic":
            self.critical_fractile = float(stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps))
        elif fractile_type in ("tau_amortized", "amortized"):
            if shelf_life_tau is not None and shelf_life_tau > 0:
                self.critical_fractile = float(
                    stockout_penalty_pu / (stockout_penalty_pu + holding_cost_h + spoilage_penalty_ps / shelf_life_tau)
                )
            else:
                self.critical_fractile = float(
                    stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps + holding_cost_h)
                )
        elif shelf_life_tau is not None and shelf_life_tau > 0:
            self.critical_fractile = float(
                stockout_penalty_pu / (stockout_penalty_pu + holding_cost_h + spoilage_penalty_ps / shelf_life_tau)
            )
        else:
            self.critical_fractile = float(
                stockout_penalty_pu / (stockout_penalty_pu + spoilage_penalty_ps + holding_cost_h)
            )

        self.z_factor = z_factor
        self.safety_buffer = 0.0
        self.sigma = 0.0

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_flat)
        preds_in_sample = self.model.predict(X_scaled)
        residuals = y_flat - preds_in_sample
        self.sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        z = float(self.z_factor) if self.z_factor is not None else float(norm.ppf(self.critical_fractile))
        self.safety_buffer = float(max(0.0, z * self.sigma))

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_flat)
        point_preds = self.model.predict(X_scaled)
        return point_preds + self.safety_buffer


SafetyStockBaseline = SafetyStockPTO


class MLPPTO(BasePTOModel):
    """Two-Stage PTO baseline 8: Deep Neural Network (MLP) trained via MSE + MILP."""

    def __init__(
        self,
        in_features: int = 14,
        hidden_dim: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        super().__init__("MLP_PTO")
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.net = DemandPredictorMLP(in_features=in_features, hidden_dim=hidden_dim)

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_flat)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_flat, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_scaled = self.scaler.transform(X_flat)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds = self.net(X_tensor).cpu().numpy()
        return preds


class KNNSAAModel(BasePTOModel):
    """Two-Stage PTO baseline 9: K-Nearest Neighbors Sample Average Approximation (KNN-SAA) + MILP."""

    def __init__(self, k_neighbors: int = 5):
        super().__init__("KNN_SAA")
        self.k = k_neighbors
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        self.y_train = None

    def _fit_flat(self, X_flat: np.ndarray, y_flat: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_flat)
        self.nn.fit(X_scaled)
        self.y_train = np.array(y_flat, dtype=np.float64)

    def _predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """Predicts the empirical conditional mean of the K nearest neighbors."""
        X_scaled = self.scaler.transform(X_flat)
        indices = self.nn.kneighbors(X_scaled, return_distance=False)
        neighbor_demands = self.y_train[indices]
        preds = np.mean(neighbor_demands, axis=1)
        return preds


KNNSAAPTO = KNNSAAModel


class TemporalSeqNetPTO(BasePTOModel):
    """
    Two-Stage PTO baseline 10: Deep Temporal Sequence Predictor (LSTM/GRU) + MILP.
    Includes learnable store entity embeddings, Softplus activation, and validation checkpointing.
    """

    def __init__(
        self,
        in_features: int = 14,
        num_stores: int = 10,
        horizon_T: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        store_embed_dim: int = 8,
        bidirectional: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        batch_size: int = 16,
        patience: int = 8,
        device: str | None = None,
    ):
        super().__init__("TemporalSeqNet_PTO")
        self.in_features = in_features
        self.num_stores = num_stores
        self.horizon_T = horizon_T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.store_embed_dim = store_embed_dim
        self.bidirectional = bidirectional
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.net = TemporalSequenceModule(
            in_features=in_features,
            num_stores=num_stores,
            horizon_T=horizon_T,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            store_embed_dim=store_embed_dim,
            bidirectional=bidirectional,
        ).to(self.device)

    def _format_X(self, X: np.ndarray) -> np.ndarray:
        """Standardizes input shape to (B, N, T, F) for recurrent processing."""
        if X.ndim == 4:
            B, s1, s2, F = X.shape
            if s1 == self.horizon_T and s2 == self.num_stores:
                return np.transpose(X, (0, 2, 1, 3))
            return X
        if X.ndim == 3:
            # (N, T, F)
            return X[np.newaxis, ...]
        if X.ndim == 2:
            num_samples, F = X.shape
            total_per_inst = self.num_stores * self.horizon_T
            if num_samples % total_per_inst == 0 and num_samples >= total_per_inst:
                B = num_samples // total_per_inst
                return X.reshape(B, self.num_stores, self.horizon_T, F)
            return X
        raise ValueError(f"Unexpected X shape: {X.shape}")

    def _format_y(self, y: np.ndarray) -> np.ndarray:
        """Standardizes target shape to (B, N, T)."""
        if y.ndim == 3:
            B, s1, s2 = y.shape
            if s1 == self.horizon_T and s2 == self.num_stores:
                return np.transpose(y, (0, 2, 1))
            return y
        if y.ndim == 2:
            # (N, T)
            return y[np.newaxis, ...]
        if y.ndim == 1:
            total_per_inst = self.num_stores * self.horizon_T
            if len(y) % total_per_inst == 0 and len(y) >= total_per_inst:
                B = len(y) // total_per_inst
                return y.reshape(B, self.num_stores, self.horizon_T)
            return y
        raise ValueError(f"Unexpected y shape: {y.shape}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """
        Fits the sequence network using MSE loss, Adam optimizer, and early stopping.
        Accepts 4D (B, N, T, F) or 2D (M, F).
        """
        X_fmt = self._format_X(np.asarray(X_train, dtype=np.float32))
        y_fmt = self._format_y(np.asarray(y_train, dtype=np.float32))

        # Fallback for arbitrary 2D flat inputs
        if X_fmt.ndim == 2:
            X_scaled = self.scaler.fit_transform(X_fmt)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_fmt, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=min(self.batch_size * 10, len(X_fmt)), shuffle=True)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = nn.MSELoss()
            self.net.train()
            for _ in range(self.epochs):
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    pred = self.net(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
            self.is_fitted = True
            return self

        B, N, T, F = X_fmt.shape
        X_flat = X_fmt.reshape(-1, F)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(B, N, T, F)

        target_mean = float(np.mean(y_fmt))
        self.net.calibrate_bias(target_mean)

        train_dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_fmt, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_fmt = self._format_X(np.asarray(X_val, dtype=np.float32))
            y_val_fmt = self._format_y(np.asarray(y_val, dtype=np.float32))
            B_v, N_v, T_v, F_v = X_val_fmt.shape
            X_val_scaled = self.scaler.transform(X_val_fmt.reshape(-1, F_v)).reshape(B_v, N_v, T_v, F_v)
            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_fmt, dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        patience_counter = 0

        for _ in range(1, self.epochs + 1):
            self.net.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.net(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= B

            if val_loader is not None:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        pred = self.net(bx)
                        val_loss += criterion(pred, by).item() * len(bx)
                val_loss /= B_v
                eval_loss = val_loss
            else:
                eval_loss = train_loss

            if eval_loss < best_loss - 1e-4:
                best_loss = eval_loss
                best_state = copy.deepcopy(self.net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.net.load_state_dict(best_state)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts demand. Preserves dimensions:
          - (B, N, T, F) -> returns (B, N, T)
          - (N, T, F) -> returns (N, T)
          - (M, F) -> returns (M,)
        """
        self.net.eval()
        X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.ndim == 2:
            total_per_inst = self.num_stores * self.horizon_T
            if len(X_arr) % total_per_inst == 0 and len(X_arr) >= total_per_inst:
                B = len(X_arr) // total_per_inst
                X_scaled = self.scaler.transform(X_arr).reshape(B, self.num_stores, self.horizon_T, self.in_features)
                bx = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    preds = self.net(bx).cpu().numpy()
                return np.maximum(0.0, preds).reshape(-1)
            # Arbitrary 2D sample count
            X_scaled = self.scaler.transform(X_arr)
            bx = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                preds = self.net(bx).cpu().numpy()
            return np.maximum(0.0, preds)

        if X_arr.ndim == 3:
            # (N, T, F)
            X_flat = X_arr.reshape(-1, self.in_features)
            X_scaled = self.scaler.transform(X_flat).reshape(1, self.num_stores, self.horizon_T, self.in_features)
            bx = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                preds = self.net(bx).cpu().numpy()
            return np.maximum(0.0, preds).squeeze(0)

        if X_arr.ndim == 4:
            B, s1, s2, _ = X_arr.shape
            if s1 == self.horizon_T and s2 == self.num_stores:
                X_perm = np.transpose(X_arr, (0, 2, 1, 3))
            else:
                X_perm = X_arr
            X_flat = X_perm.reshape(-1, self.in_features)
            X_scaled = self.scaler.transform(X_flat).reshape(B, self.num_stores, self.horizon_T, self.in_features)
            bx = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                preds = self.net(bx).cpu().numpy()
            return np.maximum(0.0, preds)

        raise ValueError(f"Unsupported shape for predict: {X_arr.shape}")


TemporalSequenceBaseline = TemporalSeqNetPTO
TemporalSequenceNetBaseline = TemporalSeqNetPTO
TemporalPTO = TemporalSeqNetPTO
