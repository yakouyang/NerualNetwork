# -*- coding: UTF-8 -*-

import numpy as np

class Sigmoid(object):
    def function(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def deviation(self, output):
        return output * (1 - output)

