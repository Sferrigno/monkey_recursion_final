import csv
import string
import os


def merge(var, where):
	loc= "%s/%s" % (os.getcwd(), where)
	relevant = set()
	for f in os.listdir(loc):
		if (var in f and ".csv" in f) and (not ("merged" in f)):
			relevant.add(f)

	relevant = list(relevant)
	txt = ""
	first = True
	for file in relevant:
		file_loc = "%s/%s" % (where, file)
		o = open(file_loc, "r")
		i = 0
		for line in o:
			if (first == True) or (i != 0):
				txt += line
			i += 1
			first = False

	out = "%s/%s%s.csv" % (where,"merged_", var)
	f = open(out, "w+")
	f.write(txt)
	f.close()









def merge_wrapper(lst, where="/"):
	for var in lst:
		merge(var, where)




if __name__ == "__main__":
	merge_wrapper(["alpha", "beta", "noise", "theta"], where="model_out")