#generates the data from the generative story

import sys
import numpy

num_items = int(sys.argv[1])
dim = int(sys.argv[2])
epsilon = float(sys.argv[3])
#q = float(  )


Z = numpy.random.randn(num_items, dim)
print Z

class Words:
    pass
X = Words();
Y = WordS();
X.words = numpy.array(xrange(num_items))

X.features = Z;
X.G = numpy.zeros((num_items,num_items),dtype=numpy.float)

