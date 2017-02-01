import csv
import re
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
import networkx as nx
from scipy.stats import fisher_exact

class data:

	"""
	A representation of a dataset for clustering and graph inferrence.
	
	=== Attributes ===
	- parameters: list of str -- relevant parameters are {"PR", "IQ", "Q2",
                                "noUF", and "SPLIT"}; efects of parameters
                                are described under the "Parameters" heading
                                below.
	- data: list of list -- matrix where each column is a situation (occurrence
				of an indefinite pronoun in the corpus), and each
				row is a language. Each cell contains the indefinite
				pronoun used in a particular language for a given
				situation.
				
	- token_index: list of list -- Sublists correspond to the situation columns
				in the data attribute. Each sublist contains 2
				items: the line number and the word index within that line
				for a given situation.
	- ontological: list of str -- The ontological categories corresponding
				to the situation columns in the data attribute.
				Ontological categories include {"body", "one", "thing",
				"where", ...}
	- annotation: list of str -- The annotated Haspelmath categories corresponding
				to the situation columns in the data attribute.
	- utterance: list of str -- The English sentence corresponding to the
				situation columns in the data attribute.

	=== Additional attributes for graph inferrence ===
	- sense_names: list -- the names of each of the functions (each function is
				a node in the graph inferred)
	- n_S: int -- the number of senses in sense_names
	- G: networkx graph -- a graph with nodes for each function in sense_names
    	- senses: list of networkx graph -- each graph corresponds to a marker
				in a particular langauge, and is a subgraph of
				the attribute G with only the nodes (functions)
				relevant for the particular marker.
    	- languages: list of str -- list of languages in data (numbered)

	=== Parameters ===
	- "PR" -- include Predicative (PRED) as a separate category. This
			group includes sentences where an indefinite pronoun
			is predicative, for example "That is *something?"
			or "It's *nothing".
	- "IQ" -- include indirect questions in the Question (QU) category.
			This means sentences like "Does he really think [she
			did *something that awful]?", where the indefinite pronoun
			is in a subordinate clause within a quesiton would be
			considered Question (QU) instead of what their primary
			marking (in this case Specific (SP)).
	- "Q2" -- include cases that are ambiguous between question structure
			and declarative structure. (The default is to leave these
			out). An example of a Q2 sentence is "You have something to
			eat?", which could be interpreted as a declarative with
			question intonation (which would be marked as Specific (SP))
			or as an elided version of "Do you have something to eat?"
			(which would be marked as QU). The annotation used is the
			majority of the votes of the annotators.
	- "noUF" -- exclude all cases that are marked as Unclear Function (UF). (The
			default is to include them as their own category).
	- "SPLIT" -- split multi-word translations of indefinite pronouns into separate
			words. This means that the shared morphology of Turkish
			'bir sey' and 'sey' would be taken into account. Otherwise
			they are treated as separate types.
	"""
	
	def __init__(self, data_path, stem_dict_path, parameters):
		"""
		(data, str, str, list) -> None

		data_path: str -- path to input tsv
		stem_dict_path: str -- path to stem_dict csv
		parameters: list of str -- list of parameters

		Initialize a data object with attributes parameters, data,
		token_index, ontolgical, annotation, and utterance.
		"""
		self.parameters = parameters
		self.read_data(data_path, stem_dict_path)

	def read_data(self, data_path, stem_dict_path):
		"""
		(data, str, str) -> None
		data_path: str -- path to input tsv
		stem_dict_path: str -- path to stem_dict csv

		Initialize the data object with attributes data, token_index,
		ontological, annotation, and utterance based on data_path
		and stem_dict_path.
		"""
		
		# read in stem_dict from stem_dict_path
		stem_dict_raw = list(csv.reader(open(stem_dict_path), delimiter = ','))[1:]
		stem_dict = { (int(l[0]), l[1]) : l[2] for l in stem_dict_raw }
		
		# initialize attributes
		self.token_index = []
		self.ontological = []
		self.annotation = []
		self.utterance = []
		
		# read in data from data_path
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
		
		# convert attrubutes to np arrays
		self.ontological = np.array(self.ontological)
		self.annotation = np.array(self.annotation)
		self.data = np.array(self.data)
		self.token_index = np.array(self.token_index)
		return

	def create_oc_mds_files(self, association = 'None'):
		"""
		(data) -> None
		
		Create files for OCMDS from data. This creates:
			- a csv distance matrix between terms in OCMDS format
			- a labels file with per-language labels for each situation
			- a gold file with (English) utterance, (English) word, annotation, and
			  ontological category
		"""
		
		# frequency cut off is done in OC script
		if association == 'None':
			legal_term_set = set([(li,t) for sit,f in enumerate(self.data)
				for li,tt in enumerate(f) for t in tt])
		else:
			legal_term_set = set([(li,t) for li,t,f in self.get_tf_associations(test = association)])
		terms = set([(li,w) for d in self.data for li,dl in enumerate(d) 
			for w in dl if (li,w) in legal_term_set])
		term_dict = { k : v for v,k in enumerate(sorted(terms)) }
		
		# initialize lg_ixx (list of list of terms, with terms for each
		# language in a separate list)
		# initialize M (distance matrix betweeen terms)
		lg_ixx = [[] for i in range(30)]
		for k,v in term_dict.items():
			lg_ixx[k[0]].append(v)
		M = np.zeros((self.data.shape[0],len(terms)), dtype = 'int') + 6
		
		# populate M
		for di,d in enumerate(self.data):
			M[di,[term_dict[(li,t)] for li,tt in enumerate(d) for t in tt if (li,t) in terms]] = 1
			for ti,t in enumerate(d):
				if len(t) == 0:
					M[di,lg_ixx[ti]] = 9

		# write M to a csv file
		fb = 'oc_%s' % ('_'.join(sorted(self.parameters)))
		with open('%s.csv' % fb, 'w') as fh:
			fh.write('sit,%s\n' %
					 ','.join('%d:%s' % k for k in
						sorted(term_dict, key = lambda k : term_dict[k])))
			for i,r in enumerate(M):
				fh.write('%d,%s\n' % (i,','.join([str(c) for c in r])))
		
		# write language labels to a csv (to color code OCMDS plots by language)
		with open('%s_labels.csv' % fb, 'w') as fh:
			fh.write('sit,%s\n' % ','.join('L%d' % li for li in range(30)))
			for i,d in enumerate(self.data):
				fh.write('%d,%s\n' % (i,','.join(' '.join(e) for e in d)))

		# write (English) utterance, (English) word, annotation, and ontological
		# category (to color code OCMDS plots)
		with open('%s_gold.csv' % fb, 'w') as fh:
			fh.write('utt,word,annotation,ontological\n')
			for o,a,ix in zip(self.ontological,
					  self.annotation,
					  self.token_index):
				fh.write('%d,%d,%s,%s\n' % (ix[0],ix[1],a,o))
				
	def create_graph_inference_objects(self,
			representation_level = 'exemplar',
			frequency_cutoff = 1):
		"""
		(data, str, int) -> None
		representation_level: str -- in {'exemplar', 'function'}; when
					this is set as 'function', functions are nodes,
					and when it's set as 'exemplar', individual
					situations (columns in data attribute)
		frequency_cutoff: int -- situations that occurr fewer than
					frequency_cutoff times are not included in
					the graph_inferrence objects.
		
		Create graph inferrence attributes (described under "Additional
		attributes for graph inferrence" in class docstring). This is
		based on ACTUAL co-occurrences of functions with terms, and is
		succeptible to data scarcity problems.
		"""
		
		#
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

	def get_tf_associations(self, test):
		# test = {not dissociated,associated}
		tf_set = set()
		# this is the set in which all Term - Function pairs will be contained
		# that cannot be dissociated (i.e, for which we do not know for sure that
		# they are not associated) - done with Fisher Exact tests
		for onto in set(self.ontological):
			if onto not in ['body','thing']: continue
			d_onto = self.data[self.ontological == onto]
			for li in range(30):
				terms = set([w for dd in d_onto for w in dd[li]])
				for term in terms:
					for annot in set(self.annotation):
						valid = False
						if annot == 'UF': continue
						d_onto_annot = self.data[(self.ontological == onto) * (self.annotation == annot)]
						aa = len([t for t in d_onto_annot if term in t[li]]) # + term + function
						ab = len(d_onto_annot) - aa # - term + function
						ba = len([t for t in d_onto if term in t[li]]) - aa # + term - function
						bb = len(d_onto) - (aa + ab + ba) # - term - function
						if test == 'not dissociated' and fisher_exact([[aa,ab],[ba,bb]],'less')[1] > .05:
							valid = True
							tf_set.add((li,term,annot))
						if test == 'associated' and fisher_exact([[aa,ab],[ba,bb]],'greater')[1] < .05:
							valid = True
							tf_set.add((li,term,annot))
						# if aa > 0: print('%s,%d,%s,%s,%r,%d,%d,%d' % (onto,li,term,annot,valid,aa,ba,ab))
		return tf_set

	def create_graph_inference_estimation(self, test = 'not dissociated'):
		"""
		(data, str, int) -> None
		test: str -- in {'not dissociated', 'associated'}; when this is
				set as 'not dissociated', for each term, all
				functions that are not dissociated with the
				term are included in that term's subgraph in
				the senses attribute. When this is set as 'associated'
				all functions that are associated with with the
				term are included in that term's subgraph in the
				senses attribute. The setting 'not dissocated'
				is more permissive.
		
		Create graph inferrence attributes (described under "Additional
		attributes for graph inferrence" in class docstring). This is
		based on EXPECTED co-occurrences of functions with terms, and is
		less succeptible to data scarcity problems.
		"""

		tf_set = self.get_tf_associations(test)
		
		# construct dict mapping language and term to a list of functions
		lang_term_to_functions = {}
		for item in sorted(tf_set):
			key = (item[0], item[1])
			if key in lang_term_to_functions:
				lang_term_to_functions[key].append(item[2])
			else:
				lang_term_to_functions[key] = [item[2]]

		self.sense_names = list(sorted(set([item[2] for item in tf_set])))
		symbols = list(range(len(self.sense_names)))
		self.senses = []

		# add subgraph for each language-marker
		for t in lang_term_to_functions:
			L = t[0]
			N = t[1]
			T = [self.sense_names.index(sn) for sn in lang_term_to_functions[t]]
			new_g = nx.Graph(language = L, term = N)
			self.senses.append(new_g)
			self.senses[-1].add_nodes_from(T)

		# construct G
		self.G = nx.Graph()
		for i in symbols:
			self.G.add_node(i, hasp_type=self.sense_names[i])
		self.languages = set([g.graph['language'] for g in self.senses])
		self.n_S = self.G.number_of_nodes()
		return


if __name__ == "__main__":
	import sys
	d = data(sys.argv[1], sys.argv[2], ["SPLIT"])
	d.create_oc_mds_files()


