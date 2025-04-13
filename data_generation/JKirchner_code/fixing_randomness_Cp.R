# this script allows to precompute input tracer data that will be used to simulate the data using the threebow model


rm(list=ls())  #clear environment

library(dplyr)

options(digits=7)
# We load an arbitrary dataset (among Pully, Lugano and Basel) just to get the length of the time series.
dat <-  read.csv("../ref_datafiles/Pully.csv")

p <- dat$p
t <- dat$t
n <- length(p)
print(n)

####################################
# here we construct a synthetic random Cp for precipitation concentrations
sincoeff = 3.09 
offset = -9.41

corr_rand <- rep(NA, n)
corr_rand[1] <- rnorm(1)
for (i in (2:n)) corr_rand[i] <- 0.8*corr_rand[i-1] + 1.5 * rnorm(1);
Cp <- sincoeff*sin(2*pi*(t-0.3)) + corr_rand + offset          # add some seasonal cycle
save(Cp, file='input_concentration.rda')
