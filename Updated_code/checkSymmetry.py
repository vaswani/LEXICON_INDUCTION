import sys

import numpy

counts = []
for line in open(sys.argv[1]):
    line = line.strip();
    counts.append(map(int,line.split()[1:]))

arr = numpy.array(counts)
print 'context feature matrix has shape', arr.shape
if (arr.transpose()==arr).prod() == 1:
    print 'the context feature matrix is symmetric'

#print print 'symmetric',(arr.transpose(1, 0, 2) == arr).all()

