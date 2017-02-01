# Something more about indefinite pronouns
Code and data for the CogSci 2017 conference paper on indefinite pronouns

Breakdown of scripts and how to use them by section of paper:

* Method
	- create_dev_test_split.py
		- usage: python create_dev_test_split.py
		- description: Splits exemplars in data/full_set.csv into a dev_set and test_set of equal sizes and writes them to data/dev_set.tsv and data/test_set.csv.
		
* Are all semantic functions equally important
	- function_frequency.r
		- run in an R environment to get the data for Table 2

* Are the functions at the right level of granularity?
	- analyze_clustering.py
		- usage: python analyze_clustering.py data/test_set.tsv data/stemming_dictionary.csv
		- dependencies: data.py
		- description: Prints a clustering summary for each parameter setting (see description of parameters in the class docstring for the data class in data.py). Each summary consists of the Adjusted Rand Score with the Haspelmath gold labels followed by a confusion matrix between Haspelmath functions and clusters found by the clustering algorithm.

* The perspective of a similarity space
	- create_oc_files.py
		- usage: python create_oc_files.py data/test_set.tsv data/stemming_dictionary.csv
		- description: restructures data as input for oc.r, and writes them to files oc_SPLIT_labels.csv, oc_SPLIT_gold.csv, and oc_SPLIT.csv. For plotting the development data, replace 'data/test_set.tsv' with 'data/dev_set.tsv'. 
	- oc.r
		- usage: Rscript oc.r
		- description: Generates OC-MDS plots based on info in oc_SPLIT_test_labels.csv, oc_SPLIT_test_gold.csv and oc_test_SPLIT.csv. For plotting People instead of Things with the oc.r script, replace 'thing' on line 17 with 'body'.
	
Gallery of examples:

People                                                 |  Things
:-----------------------------------------------------:|:------------------------------------------------------:
![](/plots/onto=body_dim=2_oc_SPLIT_test_annotations.png)  |  ![](/plots/onto=thing_dim=2_oc_SPLIT_test_annotations.png)

English Things                                        |  Slovene Things
:---------------------------------------------:|:---------------------------------------------:
![](/plots/onto=thing_dim=2_oc_SPLIT_test_en.png)  |  ![](/plots/onto=thing_dim=2_oc_SPLIT_test_sl.png)

Russian Things                                        |  Indonesian Things
:---------------------------------------------:|:---------------------------------------------:
![](/plots/onto=thing_dim=2_oc_SPLIT_test_ru.png)  |  ![](/plots/onto=thing_dim=2_oc_SPLIT_test_id.png)

German People                                        |  Croatian People
:---------------------------------------------:|:---------------------------------------------:
![](/plots/onto=body_dim=2_oc_SPLIT_test_de.png)  |  ![](/plots/onto=body_dim=2_oc_SPLIT_test_hr.png)

English People                                        |  Turkish People
:---------------------------------------------:|:---------------------------------------------:
![](/plots/onto=body_dim=2_oc_SPLIT_test_en.png)  |  ![](/plots/onto=body_dim=2_oc_SPLIT_test_tr.png)

	
