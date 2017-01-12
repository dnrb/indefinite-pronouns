library(network)
library(sna)
library(ggplot2)
library(GGally)

input_path = '/home/julia/Documents/research_winter_2017/graph_inferring_output/labels_and_edges/'
output_path = '/home/julia/Documents/research_winter_2017/graph_inferring_output/graphs/'

files <- list.files(path = input_path)
for (file in files){
  if (grepl("edge", file)){
    
     edge_file <- file
     params = substr(edge_file, 7, nchar(edge_file))
     label_file <- paste("labels_", params, sep="")
     
     label_path <- paste(input_path, label_file, sep="")
     edge_path <- paste(input_path, edge_file, sep="")
     graph_output_file = paste(substr(params, 0, nchar(params) - 4), ".pdf", sep="")
     
     vertices <- read.csv(label_path, sep=",", header=F)
     edges <- read.csv(edge_path, sep=",", header=F)
     
     net <- network.initialize(length(vertices[,1]))
     net <- add.edges(net, edges[,1], edges[,2])
     
     result<- ggnet2(net, color=vertices[,2], mode="fruchtermanreingold", size=3, palette="Set3")
     hasp_output_file = paste("hasp_coded_", graph_output_file, sep="")
     print(hasp_output_file)
     
     curr_output_path = paste(output_path, hasp_output_file, sep="")
     ggsave(curr_output_path, device=cairo_pdf)
     
     if (grepl("exemplar", file)){
       result<- ggnet2(net, color=vertices[,3], mode="fruchtermanreingold", size=3, palette="Set3")
       reftype_output_file = paste("reftype_coded_", graph_output_file, sep="")
       curr_output_path = paste(output_path, reftype_output_file, sep="")
       ggsave(curr_output_path, device=cairo_pdf)
     }
  }  
}
