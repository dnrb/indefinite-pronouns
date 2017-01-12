import networkx as nx
import csv
from collections import defaultdict as dd, Counter
import numpy as np
from scipy.special import binom
from itertools import combinations

class greedy_best_first:
    """
    An abstract class for greedy best first algorithms for graph construction.

    === Attributes ===
    dataset: dataset object
    cooccurrence: nxn list of list of int, where n = self.dataset.n_S;
        cooccurrence[i][j] stores the indices of the subgraphs in
        self.dataset.senses that correspond to markers where nodes i and j
    """

    def __init__(self, dataset):
        """
        dataset: dataset object

        Infer a graph based on dataset.
        """
        self.dataset = dataset

        # initialize SG and G
        if dataset is None: return
        SG = [g.copy() for g in dataset.senses]
        print(len(SG),len(dataset.senses))
        G = dataset.G.copy()

        # initialize self.cooccurrence and possible_edges
        self.cooccurrence = [[set([c for c,g in enumerate(SG) if i in g.nodes() and j in g.nodes()])
                              for i in G.nodes_iter()] for j in G.nodes_iter()]
        possible_edges = set([(i,j) for i in G.nodes_iter() for j in G.nodes_iter()
                              if self.possible_pair(i,j)])

        # infer graph
        ctr, Ob = 0, self.get_score(SG)
        while Ob < -0:
            print(ctr,Ob)
            ctr += 1
            max_score, max_edges = -np.inf, []
            for edge in possible_edges:
                score = self.update_score(Ob, SG, edge)
                if score > max_score: max_score, max_edges = score, [ edge ]
                elif score == max_score: max_edges.append( edge )
            max_edge = max_edges[np.random.randint(len(max_edges))]
            G.add_edge(*max_edge)
            for g in self.cooccurrence[max_edge[0]][max_edge[1]]: SG[g].add_edge(*max_edge)
            possible_edges.remove(max_edge)
            Ob = max_score
        self.G, self.SG = G,SG
        return


class angluin(greedy_best_first):
    """
    Graph-inferring object that uses the Angluin algorithm.
    """

    def get_score(self, SG):
        """
        SG: list of networkx graphs -- subgraphs for each language in datset
                    parameter of __init__ for greedy_best_first (superclass)

        Return the Angluin score of a graph based on SG.
        """
        return len(SG) - np.sum([nx.number_connected_components(g) for g in SG])

    def update_score(self, O, SG, edge):
        """
        O: int -- current Angluin score
        SG: list of networkx graphs -- subgraphs for each language in datset
                    parameter of __init__ for greedy_best_first (superclass)
        edge: list of int -- the endpoints of an edge to add to the graph

        Return the updated score that would result from adding edge to SG based
        on a current score of O.
        """
        return O + np.sum([1-nx.has_path(SG[i], edge[0], edge[1])
                           for i in self.cooccurrence[edge[0]][edge[1]]])

    def possible_pair(self, i, j):
        """
        i: int -- node id
        j: int -- node id

        Return True iff there is some subgraph in SG that has the categories
        i and j in the same grouping based on self.cooccurrence.
        """
        return i < j and len(self.cooccurrence[i][j]) > 0


class dataset:
    """
    Represents a dataset stored in regier 3-header format.

    === Attributes ===
    senses: list of networkx graphs -- each graph is a subgraph of G containing
                the set of nodes associated with a marker and langauge. The
                marker and language of a subgraph sg in senses can be accessed
                through sg.graph['language'] and sg.graph['term']
    sense_names: list of str -- stores the names of the senses associated with
                each node; the sense_name of a node in G with id i
                can be found in sense_names[i]
    G: networkx graph -- graph with nodes from input file; initialized with no edges
    n_S: int -- number of nodes in G
    """

    def __init__(self, fname, category_level = 'function',
                low_freq_threshold = 5, leave_out_uf = True,
                valid_ref_types = ['thing', 'body', 'one']):
        """
        fname: str -- filename with data in exemplar Regier format (note that
                    there should be 3 headers: the first with referent types,
                    (in this case 'one', 'thing', and 'body'), the second with
                     Haspelmath functions and the third with exemplar codes
                    in the format "line-wordIndex").
        category_level: str -- in {'function', 'exemplar'}; tells whether to
                    use the exemplar codes as sense or to use the Haspelath
                    functions in the second header to group the exemplars
                    into senses.
        low_freq_threshold: int -- must be >= 0. Markers with fewer than this
                    number of occurrences are filtered out.
        """

        lines = csv.reader(open(fname))

        self.senses = []
        self.sense_names = []
        symbols = []

        # read in headers
        ref_types = next(lines)[2:]
        functions = next(lines)[2:]
        ids = next(lines)[2:]

        # filter out UF cases and cases where the referent type is not valid
        cols_to_avoid = []
        updated_functions = []
        updated_ids = []
        updated_ref_types = []
        for j, func in enumerate(functions):
            if leave_out_uf and func == "UF":
                cols_to_avoid.append(j)
            elif ref_types[j] not in valid_ref_types:
                cols_to_avoid.append(j)
            else:
                updated_ids.append(ids[j])
                updated_ref_types.append(ref_types[j])
                updated_functions.append(func)
        functions = updated_functions
        ids = updated_ids
        ref_types = updated_ref_types

        # initialize sense_names and symbols
        if category_level == "exemplar":
            symbols = list(range(len(ids)))
            self.sense_names = ids
        else:    # category_level == "function"
            self.sense_names = list(set(functions))
            symbols = list(range(len(self.sense_names)))

        # read in the situation-langauge matrix and filter out UF cases
        lines_matrix = []
        for line in lines:
            curr_col = line[:2]
            line = line[2:]
            for i in range(len(line)):
                if i not in cols_to_avoid:
                    curr_col.append(line[i])
            lines_matrix.append(curr_col)

        # add a subgraph for each langage-marker pair to SG
        for line in lines_matrix:
            if not '1' in line: continue
            elif sum([eval(x) for x in line[2:]]) < low_freq_threshold:
                continue
            L = line[0]
            N = line[1]
            if category_level == "exemplar":
                T = [symbols[c-2] for c in range(2,len(line)) if line[c] == '1']
            else:    # category_level == "function"
                T = list(set([self.sense_names.index(functions[c - 2]) for c in range(2, len(line)) if line[c] == '1']))
            new_g = nx.Graph(language = L, term = N)
            self.senses.append(new_g)
            self.senses[-1].add_nodes_from(T)

        # initialize attributes
        self.G = nx.Graph()
        if category_level == "exemplar":
            for i in symbols:
                self.G.add_node(i, referent_type=ref_types[i], line_id=ids[i],
                hasp_type=functions[i])
        else:    # category_level == "function"
            for i in symbols:
                self.G.add_node(i, hasp_type=self.sense_names[i])
        self.languages = set([g.graph['language'] for g in self.senses])
        self.n_S = self.G.number_of_nodes()
        return

class experiment:
    """
    This class runs experiments for inferring and evaluating graphs.

    === Attributes ===
    file name: str -- path to the file containing data
    """

    def __init__(self, file_name):
        """
        file_name: str -- path to file containing data for graph construction

        Constructs an experiment object with attribute file_name.
        """
        self.file_name = file_name

    def infer_graph(self, category_level = "function", algorithm = angluin,
                    low_freq_threshold = 5, leave_out_uf = True,
                    valid_ref_types = ['thing', 'one', 'body']):
        """
        algorithm: class -- graph inference algorithm class
        category_level: str -- in {'function', 'exemplar'}; tells whether to
                    use the exemplar codes as sense or to use the Haspelath
                    functions in the second header to group the exemplars
                    into senses.
        low_freq_threshold: int -- must be >= 0. Markers with fewer than this
                    number of occurrences are filtered out.

        Infers a graph based on category_level, algorithm, and
        low_freq_threshold.
        """
        self.dataset = dataset(self.file_name, category_level = category_level,
            low_freq_threshold = low_freq_threshold, leave_out_uf = leave_out_uf,
            valid_ref_types = valid_ref_types)
        a = algorithm(self.dataset)

        return a.G, a.SG
        # returns the inferred graph and the inferred subgraphs

    def evaluate_graph(self, gold_standard_edges, category_level = "function",
        algorithm = angluin, low_freq_threshold = 5):
        """
        gold_standard_edges: list of str -- stores a list of edges in the
                    gold standard graph; note that the edges are the
                    sense_names and not the sense symbols (ints corresponding
                    to indices in sense_names)
        algorithm: class -- graph inference algorithm class
        category_level: str -- in {'function', 'exemplar'}; tells whether to
                    use the exemplar codes as sense or to use the Haspelath
                    functions in the second header to group the exemplars
                    into senses.
        low_freq_threshold: int -- must be >= 0. Markers with fewer than this
                    number of occurrences are filtered out.

        Prints an evaluation of a graph with edges gold_standard_edges based
        on algorithm and self.dataset.
        """

        # initialize dataset
        self.dataset = dataset(self.file_name, category_level = category_level,
            low_freq_threshold = low_freq_threshold)

        # convert gold_standard_edges to ints (indices in self.sense_names)
        gold_standard_edges_ints = []
        for e in gold_standard_edges:
            gold_standard_edges_ints.append([self.dataset.sense_names.index(e[0]),
                self.dataset.sense_names.index(e[1])])

        # for the subgraph for each marker
        SG = [g.copy() for g in self.dataset.senses]
        edge_pair_to_violations = {}
        for g in SG:

            # add the relevant edges in gold_standard_edges_ints
            for edge in gold_standard_edges_ints:
                if edge[0] in g and edge[1] in g:
                    g.add_edge(edge[0], edge[1])

            # update edge_pair_to_violations for each pair of nodes associated
            # with a marker that are not connected
            for combo in combinations(list(g.nodes()), 2):
                if not nx.has_path(g, combo[0], combo[1]):
                    if (combo[0], combo[1]) in edge_pair_to_violations:
                        edge_pair_to_violations[(combo[0], combo[1])].append(g.graph['language'] + "-" + g.graph['term'])
                    elif (combo[1], combo[0]) in edge_pair_to_violations:
                        edge_pair_to_violations[(combo[1], combo[0])].append(g.graph['language'] + "-" + g.graph['term'])
                    else:
                        edge_pair_to_violations[(combo[0], combo[1])] = [g.graph['language'] + "-" + g.graph['term']]
        # find the score
        a = algorithm(None)    # allows us to skip the graph inferring part and just get the score
        score = a.get_score(SG)

        # print a summary
        print(str(algorithm) + " score: " + str(score))
        for item in edge_pair_to_violations:
            edge_name = [self.dataset.sense_names[item[0]], self.dataset.sense_names[item[1]]]
            print(str(edge_name) + ": " + str(edge_pair_to_violations[item]))
        return

class file_formatter():
    """
    This class groups together several methods for formatting files for this
    program.
    """

    def __init__(self):
        pass


    def save_graph(self, G, edge_path, label_path, label_type = 'sense',
                    referent_types = None, category_level = "function"):
        """
        G: networkx graph -- graph to write to files
        edge_path: str -- path to where to store edges
        label_path: str -- path to where to store labels
        label_type: str -- only relevant when category_level is "exemplar"; should be
                            in {'sense', 'referent type'}
        referent_types:dict -- only relevant when category_level is "exemplar" and
                            label_type is "referent type"; used to group exemplars
                            into referent types
        category_level: str -- in {"function", "exemplar"}; "function" means
                            that the exemplars are grouped together; "exemplar"
                            means each exemplar is treated as its own category.

        Writes the edges and labels in G to output files at edge_path and
        label_path. If category_level is "function", the labels are all "0".

        Note: 1 is added to each of the node ids written to the files because
        the node ids are 0-indexed, but no node in network graphs in R can have
        0 as an id.
        """

        # write edges to edge_path
        with open(edge_path, "w") as edge_file:
            for e in G.edges():
                edge_file.write(str(e[0] + 1) + "," + str(e[1] + 1) + "\n")

        # find labels
        if category_level == "function":
            labels = [G.node[i]['hasp_type'] for i in range(len(G.nodes()))]
        else:    # category_level == "exemplar"
            labels = self.get_labels(G, label_type, referent_types)

        # write labels to label_path
        with open(label_path, "w") as label_file:
            for i, n in enumerate(G.nodes()):
                label_file.write(str(n + 1) + "," + labels[i] +"\n")


    def get_labels(self, G, label_type, referent_type_dict):
        """
        dataset: str -- path to file in 3-header regier format
        label_type: str -- in {"sense", "referent type"}
        referent_type: dict -- maps referent types to the label for that
                        referent type (groups "one" and "body" together; both
                        map to "person")

        Returns the labels for the exemplars in dataset based on label_type
        and referent_type_dict
        """
        nodes_sorted = list(G.nodes())
        nodes_sorted.sort()
        labels = []

        for node in nodes_sorted:
            if label_type == "referent type":
                ref_type = G.node[node]['referent_type']
                labels.append(referent_type_dict[ref_type])
            elif label_type == 'sense':
                labels.append(G.node[node]['hasp_type'])
            else:    # label_type == "both"
                ref_type = G.node[node]['referent_type']
                ref_lab = referent_type_dict[ref_type]
                hasp_lab = G.node[node]['hasp_type']
                labels.append(hasp_lab + "," + ref_lab)


        #if label_type == "referent type":
            #labels = [referent_type_dict[ref_type] for ref_type in referent_types]
        #else:    # label_type == 'sense'
            #labels = next(lines)[2:]

        return labels


    def convert_to_3_header_regier_format(self, input_path, output_path,
        hasp_type_params = []):
        """
        input_path: str -- path to a file in situation-language format
        output_path: str -- path
        ref_type_params: list -- can contain "+IQ", "+Q2", and "+PRED"; the last column
                            of the file at the input path contains "IQ", "Q2," and "PRED"
                            terms, or a combination separated by semicolons; these behave
                            as follows:
                                "+Q2": include Q2 cases in the analysis (default is to leave them out)
                                "+IQ": mark IQ cases as QU
                                "PRED": treat PRED as its own category

        Writes the data in input_path to output_path in 3-header regier format.
        """

        # read in input file
        input_file = open(input_path)
        lines = csv.reader(input_file, delimiter="\t") #used to be no specified delimiter-- default to comma
        sit_lang_matrix = []
        referent_types = []
        exemplar_ids = []
        hasp_types = []

        for line in lines:

            if "+Q2" not in hasp_type_params and "Q2" in line[-1].split(";"):
                continue
            if "+IQ" in hasp_type_params and "IQ" in line[-1].split(";"):
                hasp_types.append("QU")
            elif "+PRED" in hasp_type_params and "PRED" in line[-1].split(";"):
                hasp_types.append("PRED")
            else:
                hasp_types.append(line[3])
            exemplar_ids.append(line[0] + "-" + line[1])
            referent_types.append(line[2])
            sit_lang_matrix.append(line[5:-1]) # used to be line[4:], but the 4th index now has the english sentence
        input_file.close()

        # find language-marker pairs
        lang_to_markers = {}
        for i in range(len(sit_lang_matrix[0])):
            lang_to_markers[i] = set([sit[i] for sit in sit_lang_matrix])

        # write headers to output
        output_file = open(output_path, "w")
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(["", ""] + referent_types)
        writer.writerow(["", ""] + hasp_types)
        writer.writerow(["", ""] + exemplar_ids)

        # write a line to output for each language-marker pair
        for lang in lang_to_markers:
            for m in lang_to_markers[lang]:
                if m == "":
                    continue
                curr_row = [lang, m]
                for sit in sit_lang_matrix:
                    if sit[lang] == m:
                        curr_row.append(1)
                    else:
                        curr_row.append(0)
                writer.writerow(curr_row)

        output_file.close()


    def read_edges(self, edge_file):
        """
        edge_file: str -- path to a csv file, where each line is in the format
                        "V1, V2", where V1 and V2 are vertices in a graph.

        Returns a list of edges from edge_file.
        """
        edges = []
        lines = csv.reader(open(edge_file))
        for line in lines:
            edges.append(line)
        return edges


def graph_inferring_experiment(params):
    """
    params: dict -- maps strings to different parameter values

    Infer a graph based on params.
    """
    # infer a graph
    e = experiment(params['dataset'])
    G, SG = e.infer_graph(category_level = params['category level'],
        algorithm = eval(params['algorithm']),
        low_freq_threshold = params['low freq threshold'],
        leave_out_uf = params['leave out UF'],
        valid_ref_types = params['valid ref types'])

    # write the graph edges and labels to files
    if params['write to file'] is True:
        formatter = file_formatter()
        formatter.save_graph(G, params['edges output'],
            params['label output'], label_type = params['label type'],
            referent_types = params['referent types'],
            category_level = params['category level'])
    return

def graph_evaluating_experiment(params):
    """
    params: dict -- maps strings to different parameter values.

    Evaluates a "gold standard graph" based on a dataset.

    Note: for a graph evaluating experiment, category_level should be function.
    """

    if params['category level'] != 'function':
        print("For graph evaluating experiment, category_level should be \'function\'")
        return

    # read in edges
    gold_standard_edges = file_formatter().read_edges(params['gold standard graph'])

    # run a graph evaluating experiment (prints output)
    e = experiment(params['dataset'])
    e.evaluate_graph(gold_standard_edges,
        category_level = params['category level'],
        algorithm = eval(params['algorithm']),
        low_freq_threshold = params['low freq threshold'])
    return

def get_files_for_all_ref_type_params(input_dir, input_file, output_dir):
    """
    input_dir: str-- path to input file
    input_file: str -- name of input file
    output_dir: str -- name of directory to write output files to

    Reads in the situation-language format file from input dir at input file
    and writes files to output_dir for each combination of the haspelmath
    type parameters (+/-IQ, +/-Q2, +/-PRED).
    """

    formatter = file_formatter()
    for iq in ["+IQ", "-IQ"]:
        for q2 in ["+Q2", "-Q2"]:
            for pred in ["+PRED", "-PRED"]:
                hasp_type_params = [iq, q2, pred]
                output_file_name = ""
                for item in [iq, q2, pred]:
                    if item[0] == "+":
                        output_file_name += "plus_"
                    else:
                        output_file_name += "minus_"
                    output_file_name += item[1:]
                    if item != pred:
                        output_file_name += "_"
                output_file_name += ".csv"
                formatter.convert_to_3_header_regier_format(input_dir + input_file,
                    output_dir + output_file_name, hasp_type_params = hasp_type_params)


if __name__ == "__main__":

    import os
    from parameters import params

    output_dir = '/home/julia/Documents/research_winter_2017/graph_inferring_output/labels_and_edges/'
    input_dir = "/home/julia/Documents/research_winter_2017/dev_set_files/3_header_reg_format_dev_set/"

    for file_path in os.listdir(input_dir):

        file_path = input_dir + file_path

        for cat_lev in ['function', 'exemplar']:
            for low_freq_thresh in [0, 5, 10]:

                params['dataset'] = file_path
                params['category level'] = cat_lev
                params['low freq threshold'] = low_freq_thresh


                file_name = file_path.split("/")[-1]
                file_name = cat_lev + "_thresh" + str(low_freq_thresh) + "_" + file_name

                edge_output = output_dir + "edges_" + file_name
                label_output = output_dir + "labels_" + file_name

                params['edges output'] = edge_output
                params['label output'] = label_output

                if os.path.isfile(edge_output):
                    continue

                graph_inferring_experiment(params)



    # experiment template functions for graph inference and evaluation
    #graph_inferring_experiment(params)
    ##graph_evaluating_experiment(params)


