import sys

import numpy

counts = []
for line in open(sys.argv[1]):
    line = line.strip();
    counts.append(map(int,line.split()[1:]))

arr = numpy.array(counts)
print arr.shape
print (arr.transpose()==arr).prod()
#print print 'symmetric',(arr.transpose(1, 0, 2) == arr).all()

