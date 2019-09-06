"""
Popularity baseline
"""

import numpy as np
from spotlight.factorization._components import _predict_process_ids
from collections import defaultdict


class PopularityModel(object):

    def __init__(self):

        self.freqs = None
        self.num_items = 0
        self.outs = []

    def freq_counter(self, L):

        d = defaultdict(int)  # means default value is 0

        for x in L:
            d[x] += 1

        return d

    def fit(self, interactions, verbose=False):
        self.freqs = self.freq_counter(interactions.item_ids)
        self.num_items = interactions.num_items

        for iid in range(self.num_items):
            self.outs.append(self.freqs[iid])

        self.outs = np.array(self.outs, dtype=np.float32)

        # normalize
        self.outs /= sum(self.outs)

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
        if item_ids is None:
            return self.outs
        else:
            return self.outs[item_ids]
