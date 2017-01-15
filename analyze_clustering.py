# todo: clean up

from data import data
import sys
import numpy as np
import re
#
from sklearn.cluster import AffinityPropagation
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
#
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
#
from collections import defaultdict as dd

def get_k(clustering, depth = 10):
    last = clustering[-depth: , 2]
    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    k = acceleration_rev.argmax() + 2
    return k

def get_cluster_assignments(sim_matrix, parameters):

	algorithm = next((re.split(':',f)[1] for f in parameters if f[:4] == 'algo'), 'ap')
	# from { 'hierarchical', 'kmeans', 'ap', 'ward' }
	method = next((re.split(':',f)[1] for f in parameters if f[:6] == 'method'), 'single')
	# from {'single', 'complete', 'average'} (only relevant for hierarchical clustering)
	kMk = next((int(re.split(':',f)[1]) for f in parameters if f[:1] == 'k'), 8)
	# any integer <= the data length
	damping = next((re.split(':',f)[1] for f in parameters if f[:4] == 'damping'), 0.5)
	# only relevant for AP -- in [0.5,1]
	#
	if algorithm == 'hierarchical':
		clustering = hierarchy.linkage(sim_matrix, method)
		k = get_k(clustering, 20)
		cluster_assignments = hierarchy.fcluster(clustering, k, criterion = 'maxclust')-1
	elif algorithm == 'kmeans':
		cluster_assignments = KMeans(n_clusters = kMk).fit_predict(sim_matrix)
	elif algorithm == 'ap':
		cluster_assignments = AffinityPropagation().fit_predict(sim_matrix)
	elif algorithm == 'ward':
		clustering = hierarchy.ward(sim_matrix)
		k = get_k(clustering, 20)
		cluster_assignments = hierarchy.fcluster(clustering, k, criterion = 'maxclust')-1
	return cluster_assignments

def evaluate_clustering(cluster_assignments, gold):
	non_uf = np.where(gold != 'UF')
	cluster_assignments_sub, gold_sub = cluster_assignments[non_uf], gold[non_uf]
	return adjusted_rand_score(gold_sub, cluster_assignments_sub)

def print_confusion_matrix(cluster_assignments, gold):
	m = dd(lambda : [0 for i in range(len(set(cluster_assignments)))])
	for c,g in zip(cluster_assignments,gold):
		m[g][c] += 1
	for k,v in m.items():
		print('%s\t%s' % (k[:7], ' '.join([('%d        ' % w)[:8] for w in v])))


def write_cluster_assignments(cluster_assignments, parameters):
	with open('clustering_%s.csv' % '_'.join(sorted(parameters)),'w') as fh:
		fh.write('assignment\n%s' % '\n'.join([str(s) for s in cluster_assignments])) 

def sim(d1,d2, parameters):
	comparisons = [(e1==e2) if not 'SPLIT' in parameters else
		((len(set(e1) & set(e2))/len(set(e1) | set(e2))) if len(e1+e2) > 0 else 0)
		for e1,e2 in zip(d1,d2) 
		if ('EMPTY' in parameters or (len(e1) > 0 and len(e2) > 0))]
	return (sum(comparisons)/len(comparisons)) if len(comparisons) > 0 else 0

def get_similarity_matrix(d, parameters, oix):
	#
	data_sub = d.data[oix]
	similarity_matrix = np.ones((data_sub.shape[0],data_sub.shape[0]))
	for i,di in enumerate(data_sub):
		for j,dj in enumerate(data_sub):
			if j >= i: continue
			similarity_matrix[i,j] = similarity_matrix[j,i] = sim(di, dj, parameters)
	return similarity_matrix

def single_exp():
	data_path = sys.argv[1]
	stem_dict_path = sys.argv[2]
	parameters = sys.argv[3:]
	d = data(data_path, stem_dict_path, parameters)
	#
	onto_cat = next((re.split(':',f)[1] for f in parameters if f[:4] == 'onto'),None)
	oix = sorted(set(np.where(d.ontological == onto_cat)[0]) & set(np.where(d.annotation != 'UF')[0]))
	print(len(oix))
	return
	#
	similarity_matrix = get_similarity_matrix(d, parameters, oix)
	cluster_assignments = get_cluster_assignments(similarity_matrix, parameters)
	print(evaluate_clustering(cluster_assignments, d.annotation[oix]))
	print_confusion_matrix(cluster_assignments, d.annotation[oix])

def comparative_exp():
	data_path = sys.argv[1]
	stem_dict_path = sys.argv[2]
	parameters = ['SPLIT', 'noUF']
	d = data(data_path, stem_dict_path, parameters)
	#
	#
	clustering_algos = [(a,m,k) for a in ['hierarchical', 'ward', 'ap', 'kmeans']
			    for m in [None,'complete','average','single']
			    for k in [None,2,3,4,5,6,7,8,9,10]
			    if (m != None and k == None and a == 'hierarchical') or
			    (m == None and k != None and a == 'kmeans') or
			    (m == None and k == None and a in ['ward', 'ap'])]
	for onto_cat in ['thing', 'body']:
		parameters_i = parameters + ['onto:%s' % onto_cat]
		oix = sorted(set(np.where(d.ontological == onto_cat)[0]))
		similarity_matrix = get_similarity_matrix(d, parameters_i, oix)
		for a,m,k in clustering_algos:
			parameters_j = parameters_i + ['algo:%s' % a]
			if m != None: parameters_j.append('method:%s' % m)
			if k != None: parameters_j.append('k:%r' % k)
			print(parameters_j)
			cluster_assignments = get_cluster_assignments(similarity_matrix, parameters_j)
			print(set(cluster_assignments))
			print(evaluate_clustering(cluster_assignments, d.annotation[oix]))
			print_confusion_matrix(cluster_assignments, d.annotation[oix])
	
if __name__ == "__main__":
	comparative_exp()

