library(data.table)
library(doMC)
library(foreach)
registerDoMC(cores = )

setwd('.')
source('PaRSnIP.R')

#Get list of fasta files
fasta_files <- list.files(path='../Dataset/',pattern=".fasta")
raw_info <- unlist(strsplit(unlist(strsplit(fasta_files,"_")),".fasta"));
indices <- as.numeric(as.vector(raw_info[seq(2,4002,2)]));
fasta_files <- fasta_files[order(indices)];

#Get true prediction
true_class <- read.csv('../Dataset/true_prediction.txt',header=FALSE)

#Scratch path
SCRATCH.path <- '/export/cse/rmall/scratch/SCRATCH-1D_1.1/bin/run_SCRATCH-1D_predictors.sh'
N <- length(fasta_files);

#Create feature matrix along with true labels
feature_matrix <- foreach (i = 1:N, .inorder = TRUE, .combine = 'rbind') %dopar%
{
  file.test <- paste0('/export/cse/rmall/Protein_Solubility/Dataset/',fasta_files[i]);
  aln <- read.fasta( file.test )
  aln.ali <- aln$ali
  output_prefix <- tempfile( pattern = "tmp",
                             fileext = "" )
  temp_features <- PaRSnIP.calc.features.RSAhydro.test(aln.ali,SCRATCH.path,
                                                       output_prefix,n.cores=1);
  temp_features <- c(temp_features,as.numeric(true_class$V1[i]));
  temp_features
}

feature_matrix <- as.matrix(feature_matrix);
colnames(feature_matrix) <- NULL;
names(feature_matrix) <- NULL;
write.csv(feature_matrix,'../Results/test_features.txt',row.names = F)