library(entropy)
#
data = read.csv('data/test_set.tsv', sep = '\t')
data.sub = droplevels(data[data$source != 'excluded' & data$type %in% c('body', 'thing') & data$annotation != 'UF',])

###################
# generate tables #
###################
prop.table(xtabs(~ data.sub$type + data.sub$annotation),1)
# for split per type
prop.table(xtabs(~ data.sub$annotation))
# for overall
fisher.test(xtabs(~ data.sub$type + data.sub$annotation),workspace = 2e7)
# significance testing for difference between ontological categories

#############################
# calculate Shannon-entropy #
#############################
df = data.frame(prop.table(xtabs(~ data.sub$type + data.sub$annotation),1))
p.function.body = df[df$data.sub.type == 'body',]$Freq
p.function.thing = df[df$data.sub.type == 'thing',]$Freq
smooth = 0.000001
H.body = -sum((p.function.body+smooth) * log2(p.function.body+smooth))
H.thing = -sum((p.function.thing+smooth) * log2(p.function.thing+smooth))
