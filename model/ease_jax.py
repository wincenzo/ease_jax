from functools import partial
from typing import Optional, Self, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, device_get, jit, lax
from jax.scipy.linalg import cho_factor, cho_solve
from numpy.typing import ArrayLike
from scipy.sparse import csr_array
from sklearn.preprocessing import LabelEncoder, maxabs_scale


class EASE:
    def __init__(self,
                 users: Sequence,
                 items: Sequence,
                 scores: Optional[ArrayLike] = None
                 ) -> None:
        scores = scores or []
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.implicit = not scores
        self.users = self.user_enc.fit_transform(users)
        self.items = self.item_enc.fit_transform(items)
        self.n_users = self.user_enc.classes_.size
        self.n_items = self.item_enc.classes_.size
        values = (
            np.ones(self.users.size, dtype=bool)  # type: ignore
            if self.implicit
            else maxabs_scale(scores)
        )
        self.user_item = csr_array(
            (values, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

    @staticmethod
    @jit
    def _compute_B(G: Array, l2: float) -> Array:
        '''
        G: Gram matrix
        P: Inverse gram matrix
        B: Weights matrix (with diag(B) = 0)
        l2: regularization 
        '''
        diag_idx = jnp.diag_indices(G.shape[0])
        G = G.at[diag_idx].add(l2)
        c, lower = cho_factor(G)
        P = cho_solve((c, lower), jnp.eye(G.shape[0]))
        B = P / (-jnp.diag(P)[:, None])

        return B.at[diag_idx].set(0)

    def fit(self, l2: float = 0.5) -> None:
        G = self.user_item.T @ self.user_item
        G = jnp.asarray(G.toarray())
        self.B = self._compute_B(G, l2)

    @staticmethod
    @partial(jit, static_argnames=['k'])
    def _top_k(ui: Array, B: Array, k) -> tuple[Array, Array]:
        predictions = ui @ B
        mask = ui.astype(bool)
        predictions = jnp.where(mask, -jnp.inf, predictions)

        return lax.top_k(predictions, k)
        # return lax.approx_max_k(predictions, k)

    def predict(self, users: Sequence, k: int) -> Self:
        self.k = k
        self.users = users
        users_idx = self.user_enc.transform(users)
        ui = jnp.asarray(self.user_item[users_idx, :].toarray())
        self.top_k_scores, top_k_idx = self._top_k(ui, self.B, k)
        self.top_k_items = self.item_enc.inverse_transform(
            device_get(top_k_idx).ravel())

        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": np.repeat(self.users, self.k),
            "item_id": self.top_k_items,
            "score": device_get(self.top_k_scores).ravel(),
        })
