#generates the data from the generative story

import sys
import numpy
import scipy

class Words:
    pass

num_items = int(sys.argv[1])
dim = int(sys.argv[2])
epsilon = float(sys.argv[3])
#q = float(  )
X = Words();
X.words = numpy.array(xrange(num_items))
X.Z = numpy.random.multivariate_normal(numpy.zeros(dim),numpy.identity(dim),num_items)
print X.Z
X.G = numpy.zeros((num_items,num_items),dtype=numpy.int32)
