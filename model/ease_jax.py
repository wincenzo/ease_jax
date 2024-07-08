########### PER USER POST PREDICTIONS CALCULATION

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

        values = (
            jnp.ones(self.users.size, dtype=bool)  # type: ignore
            if self.implicit
            else jnp.asarray(self.rating_scaler.fit_transform(
                data[rating_col].to_numpy()), dtype=jnp.float32)
        )

        self.n_items = self.item_enc.classes_.size
        self.n_users = self.user_enc.classes_.size

        dtype = bool if self.implicit else np.float32
        self.user_item = csr_matrix(
            (values, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=dtype,
        )

    @staticmethod
    @jit
    def _compute_B(G: Array, reg: float) -> Array:
        diag_idx = jnp.diag_indices(G.shape[0])
        G_reg = G.at[diag_idx].add(reg)
        c, lower = cho_factor(G_reg)
        P = cho_solve((c, lower), jnp.eye(G_reg.shape[0]))
        B = P / (-jnp.diag(P))[:, None]

        return B.at[diag_idx].set(0)

    def fit(self, reg: float = 0.5) -> None:
        ui_float = self.user_item.astype(np.float32)
        G = ui_float.T.dot(ui_float)
        G_dense = jnp.asarray(G.toarray())
        self.B = self._compute_B(G_dense, reg)

    @staticmethod
    @jit
    def _predict(ui: Array, B: Array) -> Array:
        return ui @ B

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _top_k(preds: Array, mask: Array, k: int) -> tuple[Array, Array]:
        candidates = jnp.where(mask, -jnp.inf, preds)

        return lax.top_k(candidates, k)

    @partial(jit, static_argnums=(0, 4))
    def _batch_predict(self,
                       ui: Array,
                       B: Array,
                       mask: Array,
                       k: int,
                       ) -> tuple[Array, Array]:
        preds = self._predict(ui, B)

        return vmap(self._top_k, in_axes=(0, 0, None))(preds, mask, k)

    def predict(self, users: Sequence, k: int) -> Self:
        self.k = k
        self.users_pred = users
        user_idx = self.user_enc.transform(users)

        ui = jnp.asarray(self.user_item[user_idx].toarray())
        mask = jnp.asarray(ui, dtype=bool)
        self.top_k_scores, top_k_idx = self._batch_predict(
            ui, self.B, mask, k)

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
    
############# FULL PREDICTIONS PRE-CALCULATION

# from functools import partial
# from typing import Optional, Self, Sequence

# import jax.numpy as jnp
# import numpy as np
# import pandas as pd
# from jax import Array, device_get, jit, lax, vmap
# from jax.scipy.linalg import cho_factor, cho_solve
# from scipy.sparse import csr_matrix
# from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


# class EASE:
#     def __init__(self,
#                  train: pd.DataFrame,
#                  user_col: str = "user_id",
#                  item_col: str = "item_id",
#                  score_col: Optional[str] = None
#                  ) -> None:
#         self.user_enc = LabelEncoder()
#         self.item_enc = LabelEncoder()
#         self.values_enc = MaxAbsScaler()

#         self.implicit = not score_col

#         self.users = self.user_enc.fit_transform(train[user_col].to_numpy())
#         self.items = self.item_enc.fit_transform(train[item_col].to_numpy())

#         values = (
#             jnp.ones(self.users.size, dtype=bool)  # type: ignore
#             if self.implicit
#             else jnp.asarray(self.values_enc.fit_transform(
#                 train[score_col].to_numpy()), dtype=jnp.float32)
#         )

#         self.n_items = self.item_enc.classes_.size
#         self.n_users = self.user_enc.classes_.size

#         dtypes = bool if self.implicit else np.float32
#         self.ui_matrix = csr_matrix(
#             (values, (self.users, self.items)),
#             shape=(self.n_users, self.n_items),
#             dtype=dtypes,
#         )

#     @staticmethod
#     @jit
#     def _compute_B(G: Array, lambda_: float) -> Array:
#         diag_idxs = jnp.diag_indices(G.shape[0])
#         G = G.at[diag_idxs].add(lambda_)
#         c, lower = cho_factor(G)
#         P = cho_solve((c, lower), jnp.eye(G.shape[0]))
#         B = P / (-jnp.diag(P))[:, None]
#         B = B.at[diag_idxs].set(0)

#         return B

#     def fit(self, lambda_: float = 0.5) -> None:
#         ui_matrix = self.ui_matrix.astype(np.float32)
#         G = ui_matrix.T.dot(ui_matrix)
#         G = jnp.asarray(G.toarray())
#         B = self._compute_B(G, lambda_)
#         self.predictions = jnp.asarray(self.ui_matrix.dot(B))

#     @staticmethod
#     @partial(jit, static_argnums=(2,))
#     def _top_k(user_predictions: Array,
#                user_interactions: Array,
#                top_k: int,
#                ) -> tuple[Array, Array]:
#         candidate_scores = jnp.where(
#             user_interactions, -jnp.inf, user_predictions)

#         return lax.top_k(candidate_scores, top_k)

#     @partial(jit, static_argnums=(0, 3))
#     def _predict_batch(self,
#                        users_idxs: Array,
#                        ui_matrix: Array,
#                        top_k: int,
#                        ) -> tuple[Array, Array]:
#         users_predictions = self.predictions[users_idxs]
#         vectorized_predict = vmap(self._top_k, in_axes=(0, 0, None))

#         return vectorized_predict(
#             users_predictions,
#             ui_matrix,
#             top_k,
#         )

#     def predict(self, target_users: Sequence, top_k: int) -> Self:
#         self.top_k = top_k
#         self.target_users = target_users
#         target_users_idxs = jnp.asarray(self.user_enc.transform(target_users))
#         target_ui_matrix = jnp.asarray(
#             self.ui_matrix[target_users_idxs].toarray(), dtype=jnp.bool)

#         self.top_k_scores, top_k_item_idxs = self._predict_batch(
#             target_users_idxs,
#             target_ui_matrix,
#             top_k)
#         self.top_k_items = self.item_enc.inverse_transform(
#             device_get(top_k_item_idxs).ravel())

#         return self

#     def to_dataframe(self) -> pd.DataFrame:
#         return pd.DataFrame(
#             {
#                 "user_id": np.repeat(self.target_users, self.top_k),
#                 "item_id": self.top_k_items,
#                 "score": device_get(self.top_k_scores).ravel(),
#             }
#         )



