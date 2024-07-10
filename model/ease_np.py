from typing import Optional, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_array
from sklearn.preprocessing import LabelEncoder, maxabs_scale


class EASE:
    def __init__(self,
                 users: ArrayLike,
                 items: ArrayLike,
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

    def fit(self, l2: float = 0.5) -> None:
        '''
        G: Gram matrix
        P: Inverse gram matrix
        B: Weights matrix (with diag(B) = 0)
        l2: regularization
        '''
        G = self.user_item.T @ self.user_item
        G = G.toarray()
        diag_idx = np.diag_indices(G.shape[0])
        G[diag_idx] += l2
        c, lower = cho_factor(G)
        P = cho_solve((c, lower), np.eye(G.shape[0]))
        self.B = P / (-np.diag(P)[:, None])
        self.B[diag_idx] = 0

    @staticmethod
    def _top_k(ui: np.ndarray, B: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        predictions = ui @ B
        mask = ui.astype(bool)
        predictions = np.where(mask, -np.inf, predictions)
        top_k_idx = np.argpartition(predictions, -k)[:, -k:]
        top_k_scores = np.take_along_axis(predictions, top_k_idx, axis=1)

        sorted_idx = np.argsort(top_k_scores, axis=1)[:, ::-1]
        sorted_scores = np.take_along_axis(
            top_k_scores, sorted_idx, axis=1)
        sorted_idx = np.take_along_axis(
            top_k_idx, sorted_idx, axis=1)
        
        return sorted_scores, sorted_idx

    def predict(self, users: ArrayLike, k: int) -> Self:
        self.k = k
        self.users = users
        users_idx = self.user_enc.transform(users)
        ui = self.user_item[users_idx, :].toarray()
        self.top_k_scores, top_k_idx = self._top_k(ui, self.B, k)
        self.top_k_items = self.item_enc.inverse_transform(top_k_idx.ravel())

        return self

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": np.repeat(self.users, self.k),
                "item_id": self.top_k_items,
                "score": self.top_k_scores.ravel(),
            }
        )
