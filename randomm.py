"""
Random baseline
"""

import numpy as np

from spotlight.factorization._components import _predict_process_ids


class RandomModel(object):

    def __init__(self):
        pass

    def fit(self, interactions, verbose=False):
        self.num_items = interactions.num_items

    def predict(self, user_ids, item_ids=None):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.num_items,
                                                  False)

        return np.random.rand(len(item_ids))
