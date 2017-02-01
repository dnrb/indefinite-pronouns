from data import data
import sys

d = data(sys.argv[1], sys.argv[2], ["SPLIT"])
d.create_oc_mds_files(association = 'None')