import csv
import re
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
import networkx as nx

class data:

	def __init__(self, data_path, stem_dict_path, parameters):
		self.parameters = parameters
		self.read_data(data_path, stem_dict_path)

	def read_data(self, data_path, stem_dict_path):
		#
		stem_dict_raw = list(csv.reader(open(stem_dict_path), delimiter = ','))[1:]
		stem_dict = { (int(l[0]), l[1]) : l[2] for l in stem_dict_raw }
		print(sorted(stem_dict.keys()))
		#
		self.token_index = []
		self.ontological = []
		self.annotation = []
		self.utterance = []
		#
		data_raw = list(csv.reader(open(data_path), delimiter='\t'))[1:]
		self.data = []
		for di,d in enumerate(data_raw):
			secondary = re.split(';',d[-1])
			if 'PR' in self.parameters and 'pred' in secondary:
				d[2] = 'PRED'
			elif 'IQ' in self.parameters and 'iq' in secondary:
				d[2] = 'QU'
			if not 'Q2' in self.parameters and 'q2' in secondary:
				continue
			if not 'noUF' in self.parameters and d[3] == 'UF':
				continue
			self.token_index.append(tuple([int(d[0]),int(d[1])]))
			self.ontological.append(d[2])
			self.annotation.append(d[3])
			self.utterance.append(d[4])
			self.data.append([])
			for li in range(30):
				try : 
					stem = stem_dict[(li,d[li+5])]
				except KeyError: 
					stem = d[li+5]
				if 'SPLIT' in self.parameters:
					self.data[-1].append([t for t in re.split(' ', stem) if t != ''])
				else: self.data[-1].append([stem])
		#
		self.ontological = np.array(self.ontological)
		self.annotation = np.array(self.annotation)
		self.data = np.array(self.data)
		self.token_index = np.array(self.token_index)
		return

	def create_oc_mds_files(self):
		# frequency cut off is done in OC script
		terms = set([(li,w) for d in self.data for li,dl in enumerate(d) for w in dl])
		term_dict = { k : v for v,k in enumerate(sorted(terms)) }
		#
		lg_ixx = [[] for i in range(30)]
		for k,v in term_dict.items():
			lg_ixx[k[0]].append(v)
		M = np.zeros((self.data.shape[0],len(terms)), dtype = 'int') + 6
		#
		for di,d in enumerate(self.data):
			M[di,[term_dict[(ti,tt)] for ti,t in enumerate(d) for tt in t]] = 1
			for ti,t in enumerate(d):
				if len(t) == 0: 
					M[di,lg_ixx[ti]] = 9
		#
		fb = 'oc_%s' % ('_'.join(sorted(self.parameters)))
		with open('%s.csv' % fb, 'w') as fh:
			fh.write('sit,%s\n' % 
					 ','.join('%d:%s' % k for k in 
						sorted(term_dict, key = lambda k : term_dict[k])))
			for i,r in enumerate(M):
				fh.write('%d,%s\n' % (i,','.join([str(c) for c in r])))
				#
		with open('%s_labels.csv' % fb, 'w') as fh:
			fh.write('sit,%s\n' % ','.join('L%d' % li for li in range(30)))
			for i,d in enumerate(self.data):
				fh.write('%d,%s\n' % (i,','.join(' '.join(e) for e in d)))

	def create_graph_inference_objects(self,
			representation_level = 'exemplar',
			frequency_cutoff = 1):
		function_dictionary = { k : v for v,k in enumerate(sorted(set(self.annotation))) }
		if representation_level == 'exemplar':
			representation_dict = { k : k for k in range(len(self.data)) }
		elif representation_level == 'function':
			representation_dict = { k : function_dictionary[f] for k,f in enumerate(self.annotation) }
		all_terms = [(li,w) for d in self.data
					 for li,dl in enumerate(d) for w in dl]
		term_count = Counter(all_terms)
		selected_terms = sorted(t for t in set(all_terms) if term_count[t] > frequency_cutoff and t[1] != '')

		self.sense_names = list(set(representation_dict.values()))
		self.sense_names.sort()
		symbols = list(range(len(self.sense_names)))
		self.senses = []  # list of subgraphs for each

		# add subgraph for each langauge-marker 
		for t in selected_terms:
			situations = sorted(set([representation_dict[i] for i,s in
									enumerate(self.data) if t[1] in s[t[0]]]))
			L = t[0]
			N = t[1]
			T = situations
			new_g = nx.Graph(language = L, term = N)
			self.senses.append(new_g)
			self.senses[-1].add_nodes_from(T)

		# construct G
		self.G = nx.Graph()
		for i in symbols:
			if representation_level == 'exemplar':
				for i in symbols:
					self.G.add_node(i, hasp_type=self.annotation[i], referent_type=self.ontological[i])
			else:    # representation_level == 'function'
				for i in symbols:
					self.G.add_node(i, hasp_type=self.sense_names[i])
		self.languages = set([g.graph['language'] for g in self.senses])
		self.n_S = self.G.number_of_nodes()
		return
