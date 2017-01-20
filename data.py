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
		#
		self.token_index = []
		self.ontological = []
		self.annotation = []
		self.utterance = []
		#
		data_raw = list(csv.reader(open(data_path), delimiter='\t'))[1:]
		self.data = []
		for di,d in enumerate(data_raw):
			if d[3] == 'excluded': continue
			secondary = re.split(';',d[4])
			annotation = d[2]
			utt,wrd = int(d[0]),int(d[1])
			onto = d[6]
			if 'PR' in self.parameters and 'pred' in secondary: annotation = 'PRED'
			elif 'IQ' in self.parameters and 'iq' in secondary: annotation = 'QU'
			if not 'Q2' in self.parameters and 'q2' in secondary: continue
			if 'noUF' in self.parameters and annotation == 'UF': continue
			self.token_index.append((utt,wrd))
			self.ontological.append(onto)
			self.annotation.append(annotation)
			self.utterance.append(d[5])
			self.data.append([])
			for li in range(30):
				try : stem = stem_dict[(li,d[li+7])]
				except KeyError: stem = d[li+7]
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
		with open('%s_gold.csv' % fb, 'w') as fh:
			fh.write('utt,word,annotation,ontological\n')
			for o,a,ix in zip(self.ontological,
					  self.annotation,
					  self.token_index):
				fh.write('%d,%d,%s,%s\n' % (ix[0],ix[1],a,o))
				
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
		
		if representation_level == 'exemplar':
			self.sense_names = list(set(representation_dict.values()))
			symbols = list(range(len(self.sense_names)))		
		else:    # representation_level == 'function'
			self.sense_names = []
			symbols = []			
			id_to_function = {v : k for k, v in function_dictionary.items()}
			ids_sorted = sorted(id_to_function.keys())			
			for k in ids_sorted:
				symbols.append(k)
				self.sense_names.append(id_to_function[k])

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

	def create_graph_inference_estimation(self, test = 'not dissociated'): # test = {not dissociated,associated}
		tf_set = set()
		# this is the set in which all Term - Function pairs will be contained
		# that cannot be dissociated (i.e, for which we do not know for sure that
		# they are not associated) - done with Fisher Exact tests
		for onto in set(d.ontological):
			if onto not in ['body','thing']: continue
			for li in range(30):
				terms = set(tuple(dd[li]) for dd in d.data[d.ontological == onto])
				for annot in set(d.annotation):
					if annot == 'UF': continue
					d_annot = d.data[(d.ontological == onto) * (d.annotation == annot)]
					for term in terms:
						aa = len([t for t in d_annot if tuple(t[li]) == term])
						ab = len(d_annot) - aa
						ba = len([t for t in d.data[d.ontological == onto] if tuple(t[li]) == term]) - aa
						bb = len(d.data[d.ontological == onto]) - (aa + ab + ba)
						if test == 'not dissociated' and fisher_exact([[aa,ab],[ba,bb]],'less')[1] > .05:
							tf_set.add((li,term,annot))
						if test == 'associated' and fisher_exact([[aa,ab],[ba,bb]],'greater')[1] < .05:
							tf_set.add((li,term,annot))
		return
