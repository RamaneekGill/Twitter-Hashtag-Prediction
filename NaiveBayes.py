import csv
import random
import math

def readCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for line in range(len(dataset)):
		dataset[line] = [str(i) for i in dataset[line]]
	return dataset


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	dataset = readCsv(filename)
	print(dataset[0])

main()
