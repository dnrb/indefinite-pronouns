### parameters for graph inferring ###

params = {
'dataset' : '/home/julia/Documents/research_winter_2017/dev_set_files/3_header_reg_format_dev_set/minus_IQ_minus_Q2_minus_PRED.csv',   # this should have 3 headers: referent type, haspelmath category, and exemplar ID
'category level' : 'function',   # in {'function', 'exemplar'}
'algorithm' : 'angluin',
'low freq threshold' : 5,	# markers with fewer than this number of occurrences are weeded out
'leave out UF': True,	# True iff data points with UF label should be left out
'valid ref types': ['thing', 'body', 'one'],

# these parameters are for evaluating a graph
'gold standard graph' : 'haspelmath_indefpro_edges.csv',

# these parameters are about where to write the output graph to a file and how to format it
'write to file' : True,
'edges output' : '/home/julia/Documents/research_winter_2017/graph_inferring_output/test_edges.csv',
'label output' : '/home/julia/Documents/research_winter_2017/graph_inferring_output/test_labels.csv',
'label type' : 'both',    # in {'referent type', 'sense', both}
'referent types' : {'one' : 'people', 'body': 'people', 'thing':'things'},
}

