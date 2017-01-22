### parameters for graph inferring ###

params = {
'dataset' : 'data/dev_set.tsv',   # this should have 3 headers: referent type, haspelmath category, and exemplar ID
'stem dict' : 'data/stemming_dictionary.csv',
'representation level' : 'function',   # in {'function', 'exemplar'}
'algorithm' : 'angluin',
'low freq threshold' : 5,	# markers with fewer than this number of occurrences are weeded out
'data params': ['noUF'], # can include a combination of {'noUF', 'PRED', 'Q2', 'IQ'}; see data class in data.py for details

# these parameters are for evaluating a graph
'gold standard graph' : 'haspelmath_indefpro_edges.csv',

# these parameters are about where to write the output graph to a file and how to format it
'write to file' : True,
'edges output' : 'sample_edges.csv',
'label output' : 'sample_labels.csv',
'label type' : 'both',    # in {'referent type', 'sense', both}
'referent types' : {'one' : 'people', 'body': 'people', 'thing':'things'},
}

