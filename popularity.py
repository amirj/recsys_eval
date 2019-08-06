"""
Popularity baseline
"""

import numpy as np
from spotlight.factorization._components import _predict_process_ids
from collections import defaultdict


class PopularityModel(object):

    def __init__(self,
                 k=20):

        self.k = 20
        self.topk = []

    def most_common(self, L, k):
        d = defaultdict(int)  # means default value is 0

        for x in L:  # go through list
            d[x] += 1  # increment counts

        # sort dict items by value (count) in descending order
        sorted_items = sorted(d.items(), key=lambda i: i[1], reverse=True)

        # extract the keys
        sorted_keys = [k for k, v in sorted_items]

        # take k best
        return sorted_keys[:k]

    def fit(self, interactions, verbose=False):
        self.topk = self.most_common(interactions.item_ids, 20)
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
        outs = []
        for item in item_ids:
            if int(item) in self.topk:
                outs.append(float(1/(self.topk.index(int(item))+1)))
            else:
                outs.append(0.0)

        outs = np.array(outs, dtype=np.float32)

        return outs
