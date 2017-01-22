# Something more about indefinite pronouns
Code and data for the CogSci 2017 conference paper on indefinite pronouns

Breakdown of scripts and how to use them by section of paper:

* Method
	- create_dev_test_split.py
		- usage: python create_dev_test_split.py

* Are all functions equally important?
	- PLACEHOLDER

* Are the functions at the right level of granularity?
	- analyze_clustering.py
		- usage: python analyze_clustering.py
		- dependencies: data.py

* The perspective of a similarity space
	- oc.r
		- usage: Rscript oc.r

* Returning to the semantic map
	- graph_inferring.py
		- usage: python graph_inferring.py
		- dependencies: data.py, graph_parameters.py
	- draw_regier_graph.R
		- usage: Rscript draw_regier_graph.R
