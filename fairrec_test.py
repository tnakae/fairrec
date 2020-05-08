import numpy as np

from fairrec import FairRec


def get_embeddings(num_user, num_item, dim=8, std_noize=0.001):
    base_vec = np.arange(dim) / dim
    user_rotation = np.arange(num_user) / num_user
    item_rotation = np.arange(num_item) / num_item

    user_rotation[:] = user_rotation + np.random.randn(num_user) * std_noize
    item_rotation[:] = item_rotation + np.random.randn(num_item) * std_noize

    user_embs = np.sin(2. * np.pi *
        (user_rotation[:, None] + base_vec[None, :]))
    item_embs = np.sin(2. * np.pi *
        (item_rotation[:, None] + base_vec[None, :]))

    return user_embs, item_embs


def test_fairrec(num_user=10, num_item=10, k=10, alpha=1.):
    user_embs, item_embs = get_embeddings(num_user, num_item)
    model = FairRec(user_embs, item_embs, k=k, alpha=alpha)
    return model.recommend()
