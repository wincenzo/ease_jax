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
        self.user_item_matrix = csr_matrix(
            (values, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=dtypes,
        )

    @staticmethod
    @jit
    def _compute_B(G: Array, lambda_: float) -> Array:
        diag_idxs = jnp.diag_indices(G.shape[0])
        G = G.at[diag_idxs].add(lambda_)
        c, lower = cho_factor(G)
        P = cho_solve((c, lower), jnp.eye(G.shape[0]))
        B = P / (-jnp.diag(P))[:, None]
        B = B.at[diag_idxs].set(0)

        return B

    def fit(self, lambda_: float = 0.5) -> None:
        user_item = self.user_item_matrix.astype(np.float32)
        G = user_item.T.dot(user_item)
        G = jnp.asarray(G.toarray())
        B = self._compute_B(G, lambda_)
        self.predictions = jnp.asarray(self.user_item_matrix.dot(B))

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _predict_single_user(user_predictions: Array,
                             user_interactions: Array,
                             top_k: int,
                             ) -> tuple[Array, Array]:
        candidate_scores = jnp.where(
            user_interactions, -jnp.inf, user_predictions)

        return lax.top_k(candidate_scores, top_k)

    @partial(jit, static_argnums=(0, 3))
    def _predict_batch(self,
                       user_idxs: Array,
                       user_item_matrix: Array,
                       top_k: int,
                       ) -> tuple[Array, Array]:
        user_predictions = self.predictions[user_idxs]
        vectorized_predict = vmap(
            self._predict_single_user, in_axes=(0, 0, None))

        return vectorized_predict(
            user_predictions,
            user_item_matrix,
            top_k,
        )

    def predict(self, target_users: Sequence, top_k: int) -> Self:
        self.top_k = top_k
        self.target_users = target_users
        target_user_idxs = jnp.asarray(self.user_enc.transform(target_users))
        target_user_item_matrix = jnp.asarray(
            self.user_item_matrix[target_user_idxs].toarray(), dtype=jnp.bool)

        self.top_k_scores, top_k_item_idxs = self._predict_batch(
            target_user_idxs,
            target_user_item_matrix,
            top_k)
        self.top_k_items = self.item_enc.inverse_transform(
            device_get(top_k_item_idxs).ravel())

        return self

    def to_dataframe(self) -> pd.DataFrame:
        results = pd.DataFrame(
            {
                "user_id": np.repeat(self.target_users, self.top_k),
                "item_id": self.top_k_items,
                "score": device_get(self.top_k_scores).ravel(),
            }
        )

        return results
