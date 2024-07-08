from functools import partial
from typing import Optional, Self, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, device_get, jit, lax, vmap
from jax.scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


class EASE:
    def __init__(self,
                 train: pd.DataFrame,
                 user_col: str = "user_id",
                 item_col: str = "item_id",
                 score_col: Optional[str] = None
                 ) -> None:
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.values_enc = MaxAbsScaler()

        self.implicit = not score_col

        self.users = self.user_enc.fit_transform(train[user_col].to_numpy())
        self.items = self.item_enc.fit_transform(train[item_col].to_numpy())

        values = (
            jnp.ones(self.users.size, dtype=bool)  # type: ignore
            if self.implicit
            else jnp.asarray(self.values_enc.fit_transform(
                train[score_col].to_numpy()), dtype=jnp.float32)
        )

        self.n_items = self.item_enc.classes_.size
        self.n_users = self.user_enc.classes_.size

        dtypes = bool if self.implicit else np.float32
        self.user_item = csr_matrix(
            (values, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=dtypes,
        )

    @staticmethod
    @jit
    def _compute_B(G: Array, lambda_: float) -> Array:
        diag_indices = jnp.diag_indices(G.shape[0])
        G = G.at[diag_indices].add(lambda_)
        c, lower = cho_factor(G)
        P = cho_solve((c, lower), jnp.eye(G.shape[0]))
        B = P / (-jnp.diag(P))[:, None]
        B = B.at[diag_indices].set(0)

        return B

    def fit(self, lambda_: float = 0.5) -> None:
        user_item = self.user_item.astype(np.float32)
        G = user_item.T.dot(user_item).toarray()
        G = jnp.asarray(G)
        B = self._compute_B(G, lambda_)
        self.predictions = jnp.asarray(self.user_item.dot(B))

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _predict_single_user(user_prediction: Array,
                             user_mask: Array,
                             k: int,
                             ) -> tuple[Array, Array]:
        candidates = jnp.where(user_mask, -jnp.inf, user_prediction)

        return lax.top_k(candidates, k)

    @partial(jit, static_argnums=(0, 3))
    def _predict(self,
                 users_idx: Array,
                 users_user_item: Array,
                 k: int,
                 ) -> tuple[Array, Array]:
        users_prediction = self.predictions[users_idx]
        vectorized_predict = vmap(
            self._predict_single_user, in_axes=(0, 0, None))

        return vectorized_predict(users_prediction, users_user_item, k)

    def predict(self, users_pred: Sequence, k: int) -> Self:
        self.k = k
        self.users_pred = users_pred
        users_pred_idx = jnp.asarray(self.user_enc.transform(users_pred))
        user_item_pred = jnp.asarray(
            self.user_item[users_pred_idx].toarray(), dtype=jnp.bool)

        self.top_k_scores, top_k_indices = self._predict(
            users_pred_idx, user_item_pred, k)
        self.top_k_results = self.item_enc.inverse_transform(
            device_get(top_k_indices).ravel())

        return self

    def to_dataframe(self) -> pd.DataFrame:
        results = pd.DataFrame(
            {
                "user_id": np.repeat(self.users_pred, self.k),
                "item_id": self.top_k_results,
                "score": device_get(self.top_k_scores).ravel(),
            }
        )

        return results
