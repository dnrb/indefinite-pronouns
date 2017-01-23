# based on the R scripts retrieved from
# http://www.unm.edu/~wcroft/MDS.html (January 23rd, 2017)

library(oc)
library(ggplot2)

language.labs = c('ar','bg','bs','cz','da','de','el','en','es','et','fi',
  'fr', 'he', 'hr', 'hu', 'id', 'it', 'ja', 'nl', 'no', 'pl', 'pr', 'ro',
  'ru', 'sl','sr','sv','tr','vi','zh')
name <- 'oc_SPLIT'
folder <- '.'
fp <- sprintf('%s/%s.csv',folder, name)

##############
# parameters #
##############
onto = 'thing' # 'body'
# ontological category to consider
ndims = 2
# number of dimensions in the OC analysis
min.terms = 5
# minimum number of positive terms a situation must have
min.freq = 5
# minimum number of situations a term must be applicable to
parameters = sprintf('onto=%s_dim=%d', onto, ndims)

#############
# read data #
#############
original.data = read.csv(sprintf('%s_gold.csv', name))
selected.sits = original.data$ontological == onto

full.data = read.csv(fp,header=TRUE)
data = full.data[selected.sits,]
nCUTs = length(2:ncol(data))
tt = data[,-1]
names = colnames(tt)
T = (tt[,1:nCUTs])
hr = rollcall(T, yea=1, nay=6, missing=9,notInLegis=8,desc=title,) 
min.sits = (1/nrow(data)) * min.freq
# minimal percentage of situations a term may occur in positive

#############
# actual OC #
#############
result = oc(hr, dims=ndims, minvotes= min.terms,
  lop= min.sits, polarity = 1:ndims, verbose = TRUE)
df = data.frame(result$legislators[,7], row.names = 1:nrow(result$legislators))
colnames(df) = c('dim.1')
for (i in 1:ndims-1) { df[[sprintf('dim.%d',i+1)]] = result$legislators[,7+i] }
write.csv(df, sprintf('%s/%s_%s.csv', folder, name, parameters))

###############
# plot resuls #
###############
coordinates.filename = sprintf('%s/%s_%s.csv', folder, name, parameters)
coordinates = read.csv(coordinates.filename, sep = ',', header = TRUE)
labels.fp = sprintf('%s/%s_labels.csv', folder, name)
labels = read.csv(labels.fp)
data.a = labels[selected.sits,]
data.a$dim.1 = coordinates$dim.1
data.a$dim.2 = coordinates$dim.2
data.a$annotation = original.data[selected.sits,]$annotation
data.a$situations = paste(original.data[selected.sits,]$utt,
  original.data[selected.sits,]$word)

xlims = c(min(-data.a$dim.1),max(-data.a$dim.1))
ylims = c(min(-data.a$dim.2),max(-data.a$dim.2))

for (language in 2:31) {
  data.sub = data.a[data.a[,language] != '',]
  top = sort(table(data.sub[,language]), decreasing = TRUE)
  top.n = names(top[top > nrow(data.a)/100 * 2])#min_freq])
  data.subsub = droplevels(data.sub[data.sub[,language] %in% top.n,])
  q = qplot(-dim.1, -dim.2, color = data.subsub[,language], label = data.subsub[,language], data = data.subsub, geom = 'text')
  q = q + xlim(xlims)
  q = q + ylim(ylims)
  q = q + theme(axis.title.x = element_blank(), axis.title.y = element_blank())
  q = q + guides(color = (show=FALSE), size = (show= FALSE))
  ggsave(sprintf('%s/%s_%s_L%d.pdf', folder, parameters, name, language), q, height = 5.5, width = 4)
}
  
q = qplot(-dim.1, -dim.2, color = annotation, label = annotation, size = 10, data = data.a, geom = 'text') + xlim(xlims) + ylim(ylims) + theme(axis.title.x = element_blank(), axis.title.y = element_blank()) + guides(color = (show=FALSE), size = (show= FALSE))
ggsave(sprintf('%s/%s_%s_annotations.pdf', folder, parameters, name), q, height = 5, width = 3.6)
