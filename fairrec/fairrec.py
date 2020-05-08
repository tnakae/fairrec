"""Fair-Rec"""
import numpy as np
import scipy.sparse as sps
from sklearn.metrics import pairwise_distances_argmin


class FairRec:
    def __init__(self, user_embeddings, item_embeddings,
                 k=10, alpha=1.0, metric="cosine"):
        self.user_emb = user_embeddings
        self.item_emb = item_embeddings
        self.num_user = self.user_emb.shape[0]
        self.num_item = self.item_emb.shape[0]

        self.k = k
        self.alpha = alpha
        self.metric = metric

        self.assignment = None
        self.sigma = None
        self.stocks = None
        self.sigma_idx = 0

        self.min_assigned = int(np.floor(
            self.num_user * k / self.num_item * self.alpha))

    def _greedy_round_robin(self):
        """Assign items to users based on greedy-round-robin rule."""
        while True:
            # 在庫がない場合は終了
            if self.stocks.sum() == 0:
                break

            # 対象ユーザ
            user_idx = self.sigma[self.sigma_idx]

            # このユーザがすでに最大数を取得している場合は終了
            if self.assignment[user_idx].nnz >= self.k:
                break

            # このユーザが持っていない在庫のあるアイテムを持ってくる
            assigned_items = self.assignment[user_idx].nonzero()[1]
            valid_items = np.argwhere(self.stocks > 0).squeeze()
            valid_items = valid_items[~np.isin(valid_items, assigned_items)]

            # このユーザが取れるアイテム在庫がない場合は終了
            if len(valid_items) == 0:
                break

            # 一番距離が近いアイテムを探す
            target_user_emb = self.user_emb[user_idx][None, :]
            valid_items_emb = self.item_emb[valid_items]
            nearest_idx = pairwise_distances_argmin(
                target_user_emb, valid_items_emb,
                metric=self.metric)[0]
            nearest_item = valid_items[nearest_idx]

            # アイテムを割り当て
            self.assignment[user_idx, nearest_item] += 1
            self.stocks[nearest_item] -= 1

            self.sigma_idx = (self.sigma_idx + 1) % self.num_user

    def recommend(self):
        self.sigma = np.random.choice(
            self.num_user, size=self.num_user, replace=False)
        self.assignment = sps.lil_matrix(
            (self.num_user, self.num_item), dtype=np.uint8)

        self.stocks = np.repeat(self.min_assigned, self.num_item)
        self._greedy_round_robin()
        self.stocks = np.repeat(self.num_user, self.num_item)
        self._greedy_round_robin()

        return self.assignment
