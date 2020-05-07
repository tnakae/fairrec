"""Fair-Rec"""
import numpy as np
import scipy.sparse as sps
from sklearn.metrics import pairwise_distances


def pairwise_distances_argsort(X, Y, metrics='euclidian'):
    distances = pairwise_distances(X, Y, metrics)
    topk_pos = np.argsort(distances, axis=1)[:, ::-1]
    return ranking


class FairRec:
    def __init__(self, user_embeddings, item_embeddings,
                 k=10, alpha=1.0, metrics="cosine"):
        self.user_emb = user_embeddings
        self.item_emb = item_embeddings
        self.k = k
        self.alpha = alpha
        self.metrics = metrics

        # num of user, item
        self.num_user = self.user_emb.shape[0]
        self.num_item = self.item_emb.shape[0]

        # lower bound of assignment for each item
        self.E_min = np.floor(self.M * k / self.N * alpha)

    def _greedy_round_robin(self):
        """Assign items to users based on *greedy-round-robin* rule.

        Parameters:
        -----------
        """
        for i in range(self.num_user):
            user_idx = self.sigma[i]
            target_user_emb = self.user_emb[user_idx][None,:]
            distance = pairwise_distances(self.item_emb, target_user_emb)
            ranking = np.argsort(distance)[::-1]

            for item_idx in ranking:
                

        pass

    def _prepare_phase1(self):
        # 各アイテムの残個数
        self.stocks = np.array([self.E_min] * self.num_item)
        # ユーザの割り当て順
        self.sigma = np.random.choice(M, size=M, replace=False)

    def _prepare_phase2(self):
        # 各アイテムの残個数は最大割り当てとする
        self.stocks = np.array([self.k] * self.num_item)
        # ユーザの割り当て順
        self.sigma = np.random.choice(M, size=M, replace=False)

    def recommend(self):
        self.X = sps.csr_matrix(shape=[self.num_user, self.num_item])
        self._prepare_phase1()
        self._greedy_round_robin()
        self._prepare_phase2()
        self._greedy_round_robin()
