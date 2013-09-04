__author__ = 'Tomer'
from collections import OrderedDict
import copy

def add(dd1, dd2):
    dd = copy.deepcopy(dd1)

    for k1 in dd2:
        d2 = dd2[k1]

        if k1 not in dd:
            dd[k1] = dict()
        d = dd[k1]

        for k2 in d2:
            if k2 in d:
                d[k2] += d2[k2]
            else:
                d[k2] = d2[k2]
    return dd


# f is a function to apply on the inner keys.
def mapInnerKeys(DD, f):
    newDD = OrderedDict()
    for s in DD:
        v = DD[s]
        newDD[s] = dict()
        for k in DD[s]:
            newDD[s][f(k)] = v[k]
    return newDD


if __name__ == '__main__':
    dd1 = {'a': {'f':1, 'g':2}, 'b': {}, 'c': {}}
    dd2 = {'a': {'f':1, 'g':2}, 'b': {'f':1, 'g':2}, 'd': {}}
    dd = add(dd1, dd2)
    print dd
    dd = add(dd2, dd2)
    print dd


