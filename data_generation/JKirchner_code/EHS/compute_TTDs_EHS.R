#################################################################################
# EHS demo -- demonstration script for EHS ensemble hydrograph separation code
#


# Version 1.4  build 2020.07.15
# Author: James Kirchner, ETH Zurich
#
#
#
# Copyright (C) 2020 ETH Zurich and James Kirchner
# Public use of this script is permitted under GNU General Public License 3 (GPL3); for details see <https://www.gnu.org/licenses/>
#
# READ THIS CAREFULLY:
# ETH Zurich and James Kirchner make ABSOLUTELY NO WARRANTIES OF ANY KIND, including NO WARRANTIES, expressed or implied, that this software is
#    free of errors or is suitable for any particular purpose.  Users are solely responsible for determining the suitability and
#    reliability of this software for their own purposes.
#
# ALSO READ THIS:
# These scripts implement the ensemble hydrograph separation approach as presented in J.W. Kirchner, "Quantifying new water fractions
#    and transit time distributions using ensemble hydrograph separation: theory and benchmark tests", Hydrology and Earth System Sciences, 
#    23, 303-349, 2019.  
# These scripts are further described in J.W. Kirchner and J.L.A. Knapp, "Calculation scripts for ensemble hydrograph
#    separation", Hydrology and Earth System Sciences, 2020.  
# Users should cite both of these papers (the first for the method, and the second for the scripts)


setwd("../../")  # change this to wherever the data file and called scripts are


rm(list=ls())  #clear environment


source("./code_threebox_model/EHS/EHS_code.R")   #replace filename as needed for future versions



########################
# supply input filename

ls_stations = c('Pully', 'Lugano', 'Basel');
ls_modes = c('small_storage', 'large_storage');
for (idxstation in 1:length(ls_stations)){
  for (idxmode in 1:length(ls_modes)){
    for (idxquantile in 0:3){
      STATION = ls_stations[idxstation];
      
      
      MODE = ls_modes[idxmode]

      SITE = sprintf("%s_%s", STATION, MODE)
      
      input_filename <- sprintf("./%s/data/%s_weekly_EHS.txt", SITE, SITE)
      
      ############################
      # here we read the data file 
      
      data <- read.csv(input_filename)
        
      
      ##################################################################################################
      # overview of "daily demo input1.txt"
      # $ date            : Factor w/ 1826 levels "01-Apr-91","01-Apr-92",..:   # date label (not used)
      # $ year            : num  91 91 91 91 91 ...                             # decimal year (not used)
      # $ p               : num  0.01 0 0 0 0 ...                               # daily precipitation (mm/day)
      # $ q               : num  1.95 1.85 1.76 1.69 1.63 ...                   # daily streamflow (mm/day)
      # $ Cp              : num  NA NA NA NA NA ...                             # tracer concentration in precipitation (per mil)
      # $ Cq              : num  -46.3 NA -44.7 -46.4 -43.9 ...                 # tracer concentration in streamflow (per mil)
      # $ pTop20            : int  1 0 0 0 0 0 0 0 0 1 ...                      # flag for top 20% of precipitation volumes
      # $ qTop20            : int  1 0 1 0 1 0 0 1 1 1 ...                      # flag for top 20% of streamflow rates
      # $ Cp.corrupted.1pct : num  -7.53 -7.46 -11.98 -9 -12.12 ...             # tracer concentration in precipitation, with 1% of points replaced by outliers
      # $ Cq.corrupted.1pct : num  -8.56 -8.13 -8.83 -8.78 -9.38 ...            # tracer concentration in streamflow, with 1% of points replaced by outliers
      # $ Cp.corrupted.2pct : num  -7.53 -7.46 -11.98 -9 -12.12 ...             # tracer concentration in precipitation, with 2% of points replaced by outliers
      # $ Cq.corrupted.2pct : num  -8.56 -8.13 -8.83 -8.78 -9.38 ...            # tracer concentration in streamflow, with 2% of points replaced by outliers
      # $ Cp.corrupted.5pct : num  -7.53 -7.46 -11.98 -9 -12.12 ...
      # $ Cq.corrupted.5pct : num  -8.56 -8.13 -8.83 -8.78 -9.38 ...
      # $ Cp.corrupted.10pct: num  -7.53 -7.46 -11.98 -9 -12.12 ...
      # $ Cq.corrupted.10pct: num  -27.2 -8.13 -8.83 -8.78 -9.38 ...
      # $ Cp.corrupted.20pct: num  -7.53 -7.46 -11.98 -40.7 -12.12 ...
      # $ Cq.corrupted.20pct: num  -27.2 -8.13 -8.83 -8.78 -61.15 ...
      # $ Cp.corrupted.30pct: num  -7.53 -7.46 -11.98 -40.7 -35.16 ...
      # $ Cq.corrupted.30pct: num  -27.2 -8.13 -8.83 -25.78 -61.15 ...
      # $ Cp.corrupted.40pct: num  -7.53 -7.46 -11.98 -40.7 -35.16 ...
      # $ Cq.corrupted.40pct: num  -27.2 -8.13 -8.83 -25.78 -61.15 ...
      # $ Cp.corrupted.50pct: num  -7.53 -7.46 -28.32 -40.7 -35.16 ...
      # $ Cq.corrupted.50pct: num  -27.2 -8.13 -8.83 -25.78 -61.15 ...
      
      
      
      # Note that the date and year information is not used here.
      # Instead, it is assumed that the rows of the input file represent an evenly spaced set of times.
      
      # now assign these four variables to their columns in the data table
      
      p <- data$p      #precipitation water flux
      q <- data$q      #streamflow water flux


      
      if (idxquantile==0){
        Qfilter <- data$Q_quantile_0   # filter for some discharge values
      }
      if (idxquantile==1){
        Qfilter <- data$Q_quantile_1   # filter for some discharge values
      }
      if (idxquantile==2){
        Qfilter <- data$Q_quantile_2   # filter for some discharge values
      }
      if (idxquantile==3){
        Qfilter <- data$Q_quantile_3   # filter for some discharge values
      }
      
      
      # now set these options
      p_thresh <- 1  # this is the threshold precipitation rate (in P units) below which P tracer inputs will be ignored
      
      nttd <- 15   # number of time steps for TTDs
      
      # these are the lag times associated with each time step
      TTD_lag <- seq(1, nttd) - 0.75  # here we treat Q and Cq as being instantaneously measured
      # if instead they are time-averaged, then replace -0.5 with -0.75 
      # (see Kirchner, 2019, but note that here the index starts at 1 rather than 0)
      
      
      
      # Note that the analyses below assume that the default options are
      # vol_wtd = FALSE
      # robust = TRUE
      # ser_corr = TRUE
      
      
      ##################################################################################
      ##################################################################################
      ##################################################################################
      # analyses WITHOUT outliers
      
      
      # set label for output filenames
      output_label <- sprintf("./%s/save/", SITE)
      
      
      Cp <- data$Cp    #precipitation tracer concentration (or del value)
      Cq <- data$Cq    #streamflow tracer concentration (or del value)
      
      
      # n <- length(p)
      # ntrain <- 
      # p <- p[n-ntrain:n]
      # q <- q[n-ntrain:n]
      # Cq <- Cq[n-ntrain:n]
      # Cp <- Cp[n-ntrain:n]
      # Qfilter <- Qfilter[n-ntrain:n]
      
      
      ###############################################################################
      # estimate transit time distributions (TTDs)
      TTD <- EHS_TTD(Cp, Cq, p, q, p_threshold=p_thresh, m=nttd-1)

      write.table(data.frame(TTD_lag, TTD$TTD, TTD$TTD_se), paste(output_label, "EHS_global.txt", sep=""), row.names=FALSE, sep=" ")

      
      
      # estimate TTDs for top 20% of discharges
      QquantileTTD <- EHS_TTD(Cp, Cq, p, q, p_threshold=p_thresh, m=nttd-1, Qfilter=Qfilter)
      
      
      # write file of TTDs
      write.table(data.frame(TTD_lag, QquantileTTD$TTD, QquantileTTD$TTD_se), paste(output_label, paste("quantile", paste( as.character(idxquantile), ".txt", sep=""), sep=""), sep="EHS_"), row.names=FALSE, sep=" ")
      
      
      
    }
  }
}












