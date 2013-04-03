import sys
file1 = open(sys.argv[1],'r')
file2 = open(sys.argv[2],'r')
out = open(sys.argv[3],'w')

for line1 in file1:
	line2 = file2.readline()

	lst1  = line1.split()
	lst2 = line2.split()
	try:
		assert(lst1[0] == lst2[0])
	except AssertionError:
		print lst1[0], lst2[0]
		#assert(False)

	out.write("%s %s"%(line1.strip(), " ".join(lst2[1:])))
	out.write("\n")
	pass

out.close()

