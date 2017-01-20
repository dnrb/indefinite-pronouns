from data import data
import networkx as nx
from collections import Counter
import sys
import numpy as np

def relation(d1, types, sample):
	found, replaces = False, None
	for ti,typ in enumerate(types):
		di_set = set([t for s in sample for t in d1[s]])
		typ_set = set([t for s in sample for t in typ[s]])
		if di_set <= typ_set: 
			found = True
		if di_set > typ_set:
			found = True
			replaces = ti
	return found, replaces

data_path = sys.argv[1]
stem_dict_path = sys.argv[2]
ontological = sys.argv[3]
freq_threshold = int(sys.argv[4])
parameters = ['SPLIT']
d = data(data_path, stem_dict_path, parameters)

ctr = Counter([(di,t) for dd in d.data for di, ts in enumerate(dd) for t in ts])
data_ont = [d.data[i] for i in range(len(d.data)) if d.ontological[i] == ontological]
data_filtered = [tuple([tuple([t for t in ts if ctr[(li,t)] >= freq_threshold]) for li, ts in enumerate(dd)]) for dd in data_ont]

out = open('minimal_functions_%s_%d.csv' % (ontological,freq_threshold), 'w')
out.write('n.languages,n.functions,examples.per.function,functions.per.example\n')
for i in range(1,30):
	samples = set()
	while len(samples) < 30:
		samples.add(frozenset(np.random.choice(np.arange(30),size=i,replace = False)))
	for sample in samples:
		types = []
		for di,dd in enumerate(data_filtered):
			found, replaces = relation(dd,types,sample)
			if not found: types.append(dd)
			elif replaces != None: types[replaces] = dd
		out.write('%d,%d,%.3f,%.3f\n' % (i,len(types),len(data_filtered)/len(types),len(types)/len(data_filtered)))
		print(i,len(types),len(data_filtered)/len(types))
out.close()