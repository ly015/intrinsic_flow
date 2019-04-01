from __future__ import division

import numpy as np
from collections import OrderedDict


class LossBuffer():

    def __init__(self, size=1000):
        self.size = size
        self.buffer = OrderedDict()

    def clear(self):
        for k in self.buffer:
            self.buffer[k] = []
    
    def add(self, errors):
        if not self.buffer:
            for k in errors:
                self.buffer[k] = []

        for k, v in errors.iteritems():
            self.buffer[k].append(v)
            if len(self.buffer[k]) > self.size * 2:
                self.buffer[k] = self.buffer[k][-self.size::]

    def get_errors(self, clear=True):
        errors = OrderedDict()
        for k, buff in self.buffer.iteritems():
            errors[k] = np.mean(buff[-self.size::])
        # print('[loss buffer] length: %d'%(len(buff[-self.size::])))
        return errors
