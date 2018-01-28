#==================================================
#==================================================
#==================================================
# Libraries
library( bio3d )
library( stringr )
library( Interpol )
library( zoo )


#==================================================
#==================================================
#==================================================
# Functions

#==================================================
# Calculate features for test sequence
PaRSnIP.calc.features.RSAhydro.test <- function( vec.seq,
                                                 SCRATCH.path,
                                                 output_prefix,
                                                 AA = unlist( strsplit("ACDEFGHIKLMNPQRSTVWY",split = "" ) ),
                                                 SS.3 = unlist( strsplit("CEH",split = "" ) ),
                                                 SS.8 = unlist( strsplit("BCEGHIST",split = "" ) ),
                                                 n.cores = 1 )
{
  #==================================================
  # Preprocess sequence
  # Step 1: Remove all inserts ("-")
  vec.seq <- vec.seq[ vec.seq != "-" ]
  # Step 2: Convert all non-standard amino acids to X
  vec.seq[ !( vec.seq %in% AA ) ] <- "X"
  
  #==================================================
  # Sequence length
  p <- length( vec.seq )
  var.log.seq.len <- log( p )
  
  #==================================================
  # Calculate molecular weight
  # df.mw <- data.frame( read.csv( "DT_MW.csv" ) )
  df.mw <- data.frame( cbind( c( "A",
                                 "R",
                                 "N",
                                 "D",
                                 "C",
                                 "E",
                                 "Q",
                                 "G",
                                 "H",
                                 "I",
                                 "L",
                                 "K",
                                 "M",
                                 "F",
                                 "P",
                                 "S",
                                 "T",
                                 "W",
                                 "Y",
                                 "V" ),
                              c( 89.1,
                                 174.2,
                                 132.1,
                                 133.1,
                                 121.2,
                                 147.1,
                                 146.2,
                                 75.1,
                                 155.2,
                                 131.2,
                                 131.2,
                                 146.2,
                                 149.2,
                                 165.2,
                                 115.1,
                                 105.1,
                                 119.1,
                                 204.2,
                                 181.2,
                                 117.1 ) ) )
  colnames( df.mw ) <- c( "AA",
                          "MW" )
  df.mw$AA <- as.vector( df.mw$AA )
  df.mw$MW <- as.numeric( as.vector( df.mw$MW ) )
  
  
  vec.mw <- NULL
  for( i in 1:length( vec.seq ) )
  {
    if( nrow( df.mw[ df.mw$AA == vec.seq[ i ], ] ) > 0 )
    {
      vec.mw <- c( vec.mw,
                   df.mw[ df.mw$AA == vec.seq[ i ], ]$MW )
    }
  }
  # var.mw <- sum( vec.mw )
  var.mw <- log( sum( vec.mw ) )
  
  #==================================================
  # Frequency turn-forming residues
  vec.tfr <- 0
  for( i in 1:length( vec.seq ) )
  {
    if( vec.seq[ i ] %in% c( "N", "G", "P", "S" ) )
    {
      vec.tfr <- vec.tfr + 1  
    }
  }
  var.tfr <- vec.tfr / length( vec.seq )
  
  #==================================================
  # Calculate GRAVY index
  var.seq <- paste( vec.seq,
                    collapse = "" )
  var.gravy <- sum( unlist( Interpol::AAdescriptor( var.seq ) ) ) / length( vec.seq )
  
  #==================================================
  # Alipathic index
  vec.ali <- rep( 0,
                  4 )
  vec.ali[ 1 ] <- sum( vec.seq == "A" )
  vec.ali[ 2 ] <- sum( vec.seq == "V" )
  vec.ali[ 3 ] <- sum( vec.seq == "I" )
  vec.ali[ 4 ] <- sum( vec.seq == "L" )
  
  var.ali <- ( vec.ali[ 1 ] + 2.9 * vec.ali[ 2 ] + 3.9 * vec.ali[ 3 ] + 3.9 * vec.ali[ 4 ] ) / p
  
  #==================================================
  # Absolute charge
  vec.ch <- rep( 0,
                 4 )
  vec.ch[ 1 ] <- sum( vec.seq == "R" )
  vec.ch[ 2 ] <- sum( vec.seq == "K" )
  vec.ch[ 3 ] <- sum( vec.seq == "D" )
  vec.ch[ 4 ] <- sum( vec.seq == "E" )
  
  var.ch <- abs( ( ( vec.ch[ 1 ] + vec.ch[ 2 ] - vec.ch[ 3 ] - vec.ch[ 4 ] ) / p ) - 0.03 )
  
  #==================================================
  # # Amino acid frequencies
  # vec.AA.freq <- rep( NA,
  #                     length( AA ) )
  # for( i in 1:length( AA ) )
  # {
  #   vec.AA.freq[ i ] <- sum( vec.seq == AA[ i ] ) / p
  # }
  # 
  # #==================================================
  # # Dipeptide frequencies
  # df.dipep <- data.frame( expand.grid( AA,
  #                                      AA ) )
  # vec.dipep <- apply( df.dipep,
  #                     1,
  #                     function( vec )
  #                     {
  #                       return( paste( as.vector( vec ),
  #                                      collapse = "" ) )
  #                     } )
  # vec.dipep.freq <- rep( NA,
  #                        length( vec.dipep ) )
  # # for( i in 1:length( vec.dipep.freq ) )
  # # {
  # #   vec.dipep.freq[ i ] <- str_count( paste( vec.seq,
  # #                                            collapse = "" ),
  # #                                     vec.dipep[ i ] ) / length( vec.dipep )
  # # }
  # var.denominator <- length( rollapply( 1:length( vec.seq ),
  #                                       2,
  #                                       sum ) )
  # for( i in 1:length( vec.dipep.freq ) )
  # {
  #   vec.dipep.freq[ i ] <- str_count( paste( vec.seq,
  #                                            collapse = "" ),
  #                                     vec.dipep[ i ] ) / var.denominator
  # }
  # 
  # #==================================================
  # # Tripeptide frequencies
  # df.tripep <- data.frame( expand.grid( AA,
  #                                       AA,
  #                                       AA ) )
  # vec.tripep <- apply( df.tripep,
  #                      1,
  #                      function( vec )
  #                      {
  #                        return( paste( as.vector( vec ),
  #                                       collapse = "" ) )
  #                      } )
  # vec.tripep.freq <- rep( NA,
  #                         length( vec.tripep ) )
  # # for( i in 1:length( vec.tripep.freq ) )
  # # {
  # #   vec.tripep.freq[ i ] <- str_count( paste( vec.seq,
  # #                                             collapse = "" ),
  # #                                      vec.tripep[ i ] ) / length( vec.tripep )
  # # }
  # var.denominator <- length( rollapply( 1:length( vec.seq ),
  #                                       3,
  #                                       sum ) )
  # for( i in 1:length( vec.tripep.freq ) )
  # {
  #   vec.tripep.freq[ i ] <- str_count( paste( vec.seq,
  #                                             collapse = "" ),
  #                                      vec.tripep[ i ] ) / var.denominator
  # }
  
  #==================================================
  #==================================================
  #==================================================
  # SCRATCH features
  
  #==================================================
  # Run SCRATCH
  run.SCRATCH( vec.seq,
               SCRATCH.path,
               output_prefix,
               n.cores )
  
  
  #==================================================
  # 3-state secondary structure classification
  file.ss <- paste( output_prefix,
                    ".ss",
                    sep = "" )
  vec.ss <- as.vector( read.fasta( file.ss )$ali )
  vec.ss.freq <- NULL
  for( i in 1:length( SS.3 ) )
  {
    vec.ss.freq <- c( vec.ss.freq,
                      sum( vec.ss == SS.3[ i ] ) )
  }
  vec.ss.freq <- vec.ss.freq / p
  
  #==================================================
  # 8-state secondary structure classification
  file.ss8 <- paste( output_prefix,
                     ".ss8",
                     sep = "" )
  vec.ss8 <- as.vector( read.fasta( file.ss8 )$ali )
  vec.ss8.freq <- NULL
  for( i in 1:length( SS.8 ) )
  {
    vec.ss8.freq <- c( vec.ss8.freq,
                       sum( vec.ss8 == SS.8[ i ] ) )
  }
  vec.ss8.freq <- vec.ss8.freq / p
  
  #==================================================
  # Solvent accessibility prediction at 0%-95% thresholds
  file.acc.20 <- paste( output_prefix,
                        ".acc20",
                        sep = "" )
  vec.acc.20.raw <- scan( file.acc.20,
                          what = "character" )
  vec.acc.20 <- as.numeric( vec.acc.20.raw[ 2:length( vec.acc.20.raw ) ] )
  vec.thresh <- seq( 0, 95, 5 )
  # vec.acc.20.final <- NULL
  # for( i in 1:length( vec.thresh ) )
  # {
  #   vec.acc.20.final <- c( vec.acc.20.final,
  #                          sum( vec.acc.20 == vec.thresh[ i ] ) / p )
  # }
  vec.acc.20.final <- sum( vec.acc.20 == vec.thresh[ 1 ] ) / p
  for( i in 2:length( vec.thresh ) )
  {
    vec.acc.20.final <- c( vec.acc.20.final,
                           sum( vec.acc.20 >= vec.thresh[ i ] ) / p )
  }
  
  #==================================================
  # Solvent accessibility prediction at 0%-95% thresholds coupled with average hydrophobicity
  ind.thresh <- which( vec.acc.20 == vec.thresh[ 1 ] )
  if( length( ind.thresh ) == 0 )
  {
    vec.rsa.hydro <- 0
  } else
  {
    vec.rsa.hydro <- ( sum( vec.acc.20 == vec.thresh[ 1 ] ) / p ) *  
      mean( unlist( Interpol::AAdescriptor( vec.seq[ ind.thresh ] ) ) )
  }
  for( i in 2:length( vec.thresh ) )
  {
    ind.thresh <- which( vec.acc.20 >= vec.thresh[ i ] )
    if( length( ind.thresh ) == 0 )
    {
      vec.rsa.hydro <- c( vec.rsa.hydro,
                          0 )
    } else
    {
      vec.rsa.hydro <- c( vec.rsa.hydro,
                          ( sum( vec.acc.20 >= vec.thresh[ i ] ) / p ) *
                            mean( unlist( Interpol::AAdescriptor( vec.seq[ ind.thresh ] ) ) ) )  
    }
  }
  
  
  #==================================================
  #==================================================
  #==================================================
  # Return feature vector
  vec.features <- c( var.log.seq.len,
                     var.mw,
                     var.tfr,
                     var.gravy,
                     var.ali,
                     var.ch,
                     vec.AA.freq,
                     vec.dipep.freq,
                     vec.tripep.freq,
                     vec.ss.freq,
                     vec.ss8.freq,
                     vec.acc.20.final,
                     vec.rsa.hydro )
  
  return( vec.features )
}



#==================================================
# Run SCRATCH
run.SCRATCH <- function( vec.seq,
                         SCRATCH.path,
                         output_prefix,
                         n.cores = 1 )
{
  # Sequence file (FASTA format)
  file.fasta <- paste( output_prefix,
                       ".fasta",
                       sep = "" )
  # Save sequence in a tmp fasta file
  write.fasta( ids = output_prefix,
               seqs = vec.seq,
               file = file.fasta )
  
  # Start running SCRATCH
  print( system2( SCRATCH.path,
                  args = c( file.fasta,
                            output_prefix,
                            n.cores ) ) )
}




#==================================================
#==================================================
# Main PaRSnIP function

PaRSnIP <- function( file.test,
                     SCRATCH.path,
                     file.output )
{
  
  # Load test sequence in fasta format
  print( "==================================================" )
  print( "Load test sequence in fasta format" )
  
  aln <- read.fasta( file.test )
  aln.ali <- aln$ali
  
  #==================================================
  # Calculate features for test sequence
  print( "==================================================" )
  print( "Calculate features for test sequence" )
  
  output_prefix <- tempfile( pattern = "tmp",
                             fileext = "" )
  
  vec.features <- PaRSnIP.calc.features.RSAhydro.test( aln.ali,
                                                       SCRATCH.path,
                                                       output_prefix,
                                                       n.cores = 32 )
  # Save features
  file.features <- paste( output_prefix,
                          "_features.txt",
                          sep = "" )
  write( vec.features,
         ncolumns = length( vec.features ),
         file = file.features )
  
  #==================================================
  # Variables h2o and h2o model
  print( "==================================================" )
  print( "Load h2o and h2o model" )
  
  # Load h2o
  h2o.init()
  
  # Load model
  gbm.model <- h2o.loadModel( path = file.h2o.model )
  
  
  #==================================================
  # Load test data and predict solubility
  print( "==================================================" )
  print( "Load test data" )
  
  MC.path <- file.path(file.features);
  
  data.hex <- h2o.importFile( path = MC.path,
                              destination_frame = "MC.hex" )
  
  #==================================================
  # Predict
  print( "==================================================" )
  print( "Predict solubility" )
  
  y.predict.raw <- h2o.predict( gbm.model,
                                data.hex )
  var.raw <- as.numeric( as.vector( y.predict.raw$p1 ) )
  metrics_df <- gbm.model@model$training_metrics@metrics$max_criteria_and_metric_scores
  threshold <- metrics_df[ metrics_df$metric == "max accuracy", ]$threshold
  var.prob <- round( get.probabilities( var.raw,
                                        threshold ),
                     4 )
  
  #==================================================
  # Save prediction values
  print( "==================================================" )
  print( "Save prediction values" )
  
  write( paste( "Probabililty to be soluble:",
                var.prob ),
         file = file.output )
}



#==================================================
#==================================================
#==================================================
# Main

#==================================================
# Command line arguments
file.test <- commandArgs()[ 3 ]
SCRATCH.path <- commandArgs()[ 4 ]
file.output <- commandArgs()[ 5 ]
no_cores <- commandArgs()[ 6 ]
if( is.na( file.output ) )
{
  file.output <- "result.txt"
}

#==================================================
# Run PaRSnIP
PaRSnIP( file.test,
         SCRATCH.path,
         file.h2o.model,
         file.output )