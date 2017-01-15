import csv

all_ips = list(csv.reader(open('data/full_set.csv'),delimiter ='\t'))
all_ips_sorted = sorted(all_ips[1:], key = lambda k : (k[6],k[2],k[4],k[3]))
dev = open('data/dev_set.tsv', 'w')
test = open('data/test_set.tsv', 'w')
dev.write('%s\n' % '\t'.join(all_ips[0]))
test.write('%s\n' % '\t'.join(all_ips[0]))
for ai,a in enumerate(all_ips_sorted):
    if ai % 2 == 0: dev.write('%s\n' % '\t'.join(a))
    else: test.write('%s\n' % '\t'.join(a))
dev.close()
test.close()
