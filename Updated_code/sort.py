#sort files.py

import sys

inputFile = sys.argv[1]
outputFile = sys.argv[2]

fRead = open(inputFile, 'r')
fWrite = open(outputFile, 'w')

lst = []
for line in fRead:
	lst.append(line.strip())
	pass

lst.sort()

for lstElem in lst:
	fWrite.write("%s\n" %(lstElem,))
	pass

fWrite.close()
fRead.close()
