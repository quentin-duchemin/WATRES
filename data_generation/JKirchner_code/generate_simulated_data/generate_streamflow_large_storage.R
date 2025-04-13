

# this script demonstrates the Rcpp implementation of the three-box model
setwd("../../") 

rm(list=ls())  #clear environment

library(data.table)
library(dplyr)

library(Rcpp)  #load Rcpp


sourceCpp("./JKirchner_code/code_threebox_model/threebox_light_v1.0_for_Rcpp.cpp")  #this is the benchmark model code (water flux module, slimmed down for parameter optimization)
sourceCpp("./JKirchner_code/code_threebox_model/threebox_v1.15_for_Rcpp.cpp")  #this is the benchmark model code (water flux module)
sourceCpp("./JKirchner_code/code_threebox_model/threebox_tracers_v1.0_for_Rcpp.cpp")  #this is the benchmark model code (solute tracer module)
sourceCpp("./JKirchner_code/code_threebox_model/threebox_age_tracking_perso_correct.cpp")  #this is the benchmark model code (age tracking module)



options(digits=7)
STATIONS = c('Lugano','Pully','Basel')
for (STATION in STATIONS){
    if (STATION=='Lugano'){
      dat <-  read.csv("./ref_datafiles/Lugano.csv")    #this is the test data file
    }
    if (STATION=='Pully'){
      dat <-  read.csv("./ref_datafiles/Pully.csv")    #this is the test data file
    }
    if (STATION=='Basel'){
      dat <-  read.csv("./ref_datafiles/Basel.csv")    #this is the test data file
    }
    
    pr <- paste0(STATION, "_large_storage/data/", collapse=NULL)
    dir.create(pr, showWarnings = FALSE);
    path_save = paste0(pr, collapse=NULL);
    
    t <- dat$t
    
    p <- dat$J
    if (is.null(p)) p <- dat$p   # to handle different column headers
    
    pet <-  dat$pet
    if (is.null(pet)) pet <- dat$PET  # to handle different column headers
    
    
    
    RMS <- function(x, wt=rep(1, length(x))) {  # root-mean-square average
      return(sqrt(weighted.mean(x*x, w=wt, na.rm=TRUE)))
    }
    
    weighted_se <- function(x, wt=rep(1, length(x))) {  # weighted standard error
      xbar <- weighted.mean(x, wt, na.rm=TRUE)
      std <- sqrt( weighted.mean( (x-xbar)*(x-xbar), wt, na.rm=TRUE) )
      n <- sum(!is.na(x*wt))
      return(std/sqrt(n))
    }
    
    dt <- 1
    agg <- 1
    n <- length(p)
    
    # parameter values
    
    if (STATION=='Lugano'){ # using calibrated parameters from site with GISID 81
      bu <- 15.      # upper box drainage exponent
      bl <- 20          # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 500     # upper box reference storage
      sl_ref <- 2500   # lower box reference storage
      sc_ref <- 2.5     # channel box storage
      fw <- 1.13          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.05      # fraction of discharge in reference state from overland flow
      f_SS <- 0.45-f_OF       # fraction of discharge in reference state from shallow subsurface flow
    }
    if (STATION=='Pully'){ # using calibrated parameters from site with GISID 81
      bu <- 15.      # upper box drainage exponent
      bl <- 20          # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 500     # upper box reference storage
      sl_ref <- 2500   # lower box reference storage
      sc_ref <- 2.5     # channel box storage
      fw <- 1.13          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.05      # fraction of discharge in reference state from overland flow
      f_SS <- 0.45-f_OF       # fraction of discharge in reference state from shallow subsurface flow
    }
    if (STATION=='Basel'){ # using calibrated parameters from site with GISID 81
      bu <- 15.      # upper box drainage exponent
      bl <- 20          # lower box drainage exponent
      bc <- 1.5          # channel box drainage exponent (FIXED at 1.5 by hydraulic geometry)
      su_ref <- 500     # upper box reference storage
      sl_ref <- 2500   # lower box reference storage
      sc_ref <- 2.5     # channel box storage
      fw <- 1.13          # fraction of upper box storage over which ET responds to storage level
      pet_mult <- 0.8   # PET multiplier
      f_OF <- 0.05      # fraction of discharge in reference state from overland flow
      f_SS <- 0.45-f_OF       # fraction of discharge in reference state from shallow subsurface flow
    }
    
    # note bc is fixed at 1.5 by hydraulic geometry
    # Note also that bu and su_ref will be nearly proportional to one another
    # Should change optimization routine so that parameters are more nearly orthogonal
    
    ##################################
    ## Benchmark hydrological model, WITHOUT parameter optimization
    
    npert <- 100        # number of perturbation time steps
    nttd <- 365*24      # number of transit time distribution time steps

    spinup <- 0         # number of time steps during spinup period
    
    
    first <- spinup+1   # for removing spinup period from results, so that further analyses are done on "first" to "last"
    last <- n
    
    
    r_et <- 1 # 0.0
    epsilon_et <- -20.0
    isotopic <-  FALSE # TRUE 
    
    load('./JKirchner_code/input_concentration.rda')
    Cp[(p<=0.0)] <- NA    
    
    
    
    ############################################
    # prepare the input for Rcpp as a named list
    
    input <- list( bu = bu,  #first seven entries are model parameters
                   bl = bl,
                   bc = bc,
                   su_ref = su_ref,
                   sl_ref = sl_ref,
                   sc_ref = sc_ref,
                   fw = fw,
                   f_OF = f_OF,
                   f_SS = f_SS,
                   r_et = r_et,
                   epsilon_et = epsilon_et,
                   isotopic = isotopic,
                   nttd = nttd,
                   npert = npert,       #number of perturbation time steps
                   agg = agg,
                   pet_mult = pet_mult,
                   n = n,   
                   dt = dt,
                   tol = 0.001,         # tolerance for numerical integration
                   t = t,               # time markers from input file
                   p = p,               # precip time series from input file
                   pet = pet,           # potential ET from input file
                   Cp = Cp)             # precip tracer concentration from input file
    
    
    
    
    ######################################################
    # now run the benchmark model, including perturbations
    bench <- c(input, threebox_model( input ))
    bench <- c(bench, threebox_tracers( bench ) ) #do tracer calculations and join results to benchmark data set
    
    filename = paste0(path_save, "simulated.txt", collapse=NULL);
    # with(bench, fwrite(data.frame(t, p, pet, et, R, qa, qOFa, qSSa, qGWa, q, qOF, qSS, qGW, su, sl, sc), filename, row.names=FALSE, na="", sep="\t"))
    with(bench, fwrite(data.frame(t, p, pet, et, R, q, Cp, Cq, qSSa, qOFa, qGWa), filename, row.names=FALSE, na="", sep="\t"))






    if (TRUE){
    
    
            
        #########################################
        # Benchmark age tracking
        nstartsave =  n-365*24*3-nttd;
        nstartsavep = nstartsave+1;
        nstart = nstartsave-nttd-100;
        nstartp = nstart+1;
        
        bench['nstart'] = nstart;
        bench['nstartsave'] = nstartsave;
        
        bench <- c(bench, threebox_age_tracking( bench ) ) # do age tracking
        
        
        
        
        
        TTD = bench$TTD[nttd:(n-nstartsave),]
        save(TTD, file= paste0(path_save, "TTD.rda", collapse=NULL))
        
        # library('arrow')
        # parquet = tempfile(fileext=".parquet")
        # write_parquet(bench$TTD, sink = parquet)
        
        
        
        # To compute the rank storage
        TTD_Su = bench$TTD_Su[nttd:(n-nstartsave),]
        save(TTD_Su, file= paste0(path_save, "TTD_Su.rda", collapse=NULL))
        
        # library('arrow')
        # parquet = tempfile(fileext=".parquet")
        # write_parquet(bench$TTD_Su, sink = parquet)
        
        # To compute the rank storage
        TTD_Sl = bench$TTD_Sl[nttd:(n-nstartsave),]
        save(TTD_Sl,  file= paste0(path_save, "TTD_Sl.rda", collapse=NULL))
        
        # library('arrow')
        # parquet = tempfile(fileext=".parquet")
        # write_parquet(bench$TTD_Sl, sink = parquet)
        
        
        # rank_storage = bench$rank_storage[nttd:(n-nstartsave),]
        # save(rank_storage, file= paste0(STATION, "_rank_storage.rda", collapse=NULL))

    }



}    
    
    

