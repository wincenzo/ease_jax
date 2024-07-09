from functools import partial
from typing import Optional, Self, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, device_get, jit, lax
from jax.scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


class EASE:
    def __init__(self,
                 data: pd.DataFrame,
                 user_col: str = "user_id",
                 item_col: str = "item_id",
                 rating_col: Optional[str] = None
                 ) -> None:
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.rating_scaler = MaxAbsScaler()
        self.implicit = not rating_col

        self.users = self.user_enc.fit_transform(data[user_col].to_numpy())
        self.items = self.item_enc.fit_transform(data[item_col].to_numpy())
        self.n_users = self.user_enc.classes_.size
        self.n_items = self.item_enc.classes_.size

        values = (
            jnp.ones(self.users.size, dtype=bool)  # type: ignore
            if self.implicit
            else jnp.asarray(self.rating_scaler.fit_transform(
                data[rating_col].to_numpy()), dtype=jnp.float32)
        )

        self.user_item = csr_matrix(
            (values, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

    @staticmethod
    @jit
    def _compute_B(G: Array, l2: float) -> Array:
        diag_idx = jnp.diag_indices(G.shape[0])
        G = G.at[diag_idx].add(l2)
        c, lower = cho_factor(G)
        P = cho_solve((c, lower), jnp.eye(G.shape[0]))
        B = P / (-jnp.diag(P))[:, None]

        return B.at[diag_idx].set(0)

    def fit(self, l2: float = 0.5) -> None:
        G = self.user_item.T @ self.user_item
        G = jnp.asarray(G.toarray())
        self.B = self._compute_B(G, l2)

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _top_k(ui: Array, B: Array, k) -> tuple[Array, Array]:
        predictions = ui @ B
        mask = ui.astype(bool)
        filtered = jnp.where(mask, -jnp.inf, predictions)

        # return lax.top_k(filtered, k)
        return lax.approx_max_k(filtered, k)

    def predict(self, users: Sequence, k: int) -> Self:
        self.k = k
        self.users_pred = users

        users_idxs = self.user_enc.transform(users)
        ui_pred = jnp.asarray(self.user_item[users_idxs].toarray())
        self.top_k_scores, top_k_idx = self._top_k(ui_pred, self.B, k)
        self.top_k_items = self.item_enc.inverse_transform(
            device_get(top_k_idx).ravel())

        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": np.repeat(self.users_pred, self.k),
                "item_id": self.top_k_items,
                "score": device_get(self.top_k_scores).ravel(),
            }
        )
