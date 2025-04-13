
// tracer concentration simlations for
// three-box conceptual hydrological model 
// for benchmark testing of ensemble hydrograph separation


// this version is designed to be called from R using Rcpp...
// NOT for stand-alone use!


// version 1.0  Build 2020.09.06
// Author: James Kirchner, ETH Zurich
//
// Copyright 2020 ETH Zurich and James Kirchner
// Public use allowed under GNU Public License v. 3 (see http://www.gnu.org/licenses/gpl-3.0.en.html or the attached license.txt file)
//
// READ THIS CAREFULLY:
// ETH Zurich and James Kirchner make ABSOLUTELY NO WARRANTIES OF ANY KIND, including NO WARRANTIES, expressed or implied, that this software is 
//    free of errors or is suitable for any particular purpose.  Users are solely responsible for determining the suitability and 
//    reliability of this software for their own purposes.


// The calculations here follow the general scheme outlined in
// Kirchner, J.W., Quantifying catchment response to precipitation, in models and field data, 
// using effect tracking and ensemble rainfall-runoff analysis  


// To run this code, users will need to install the Rcpp library.
// Windows users will also need to install Rtools; see https://cran.r-project.org/bin/windows/Rtools/





#include <Rcpp.h>
using namespace Rcpp;

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>





void model_tracers(int n,
                   NumericVector p,
                   NumericVector Cp,
                   NumericVector et,
                   bool isotopic,
                   double epsilon_et,
                   double r_et,
                   NumericVector qOF, NumericVector qSS, NumericVector qGW, 
                   NumericVector qOFa, NumericVector qSSa, NumericVector qGWa, NumericVector R, NumericVector qa,
                   NumericVector su, NumericVector sl, NumericVector sc,
                   double su_ref, double sl_ref, double sc_ref,
                   NumericVector &Cq, NumericVector &Cu, NumericVector &Cl, 
                   NumericVector &Cqa, NumericVector &Cua, NumericVector &Cla
)
  


// calculates tracer concentrations in upper and lower boxes, and streamflow, for two-box model

//receives as input:
//  n: length of time series
//  dt: 
//	p: precipitation rate time series of length n
//  Cp: precipitation concentration time series
//  et: actual evapotranspiration time series of length n
//  r_et: evapoconcentration ratio (a solute that is completely excluded from the ET flux has r_et=0)
//      Note that a tracer that is unaffected by ET will have epsilon_et=0 and/or r_et=1
//      Note also that at present, we only account for *either* fractionation *or* evapoconcentration, but not both
//	su_ref, sl_ref, sc_ref: initial volumes for upper, lower, and channel boxes
//	dt: time step
//  su, sl, sc: storage time series (instantaneous s at end of each time interval) for upper, lower, and channel boxes


//returns as output:
//  Cq, Cu, Cl: tracer concentrations in discharge, upper box, and lower box (instantaneous values at the end of each time step)
//  Cqa, Cua, Cla: tracer concentrations in discharge, upper box, and lower box (values averaged over each time step)

//	note: Cq, Cu, and Cl must be externally defined and allocated at length [n]


#define normal_formula(a, b, y0, c, h, h_plus_ct)  ( (a)/(b) + ((y0)-(a)/(b))*pow((h)/(h_plus_ct), (b)/(c)) )

#define normal_avg_formula(a, b, y0, c, h, h_plus_ct, dt) (  (a)/(b) + ((y0)-(a)/(b)) * ((h)/((b)-(c))/(dt)) * ( 1.0 - pow((h)/(h_plus_ct), (b)/(c) - 1.0)  )  )

#define exponential_formula(a, b, y0, h, dt) ( (a)/(b) + ((y0)-(a)/(b))*exp(-(b)*(dt)/(h)) )

#define exponential_avg_formula(a, b, y0, h, dt) (  (a)/(b) + ((y0)-(a)/(b)) * ((h)/(b)/(dt)) * ( 1.0 - exp(-(b)*(dt)/(h)) ) )

#define special_formula(a, y0, c, h, h_plus_ct)  ( (y0) - (a)/(c) * log((h)/(h_plus_ct)) ) 

#define special_avg_formula(a, y0, c, h, h_plus_ct, dt)  ( (y0) - (a)/(c) * (1 + ((h_plus_ct)/((c)*(dt))) * log((h)/(h_plus_ct))) ) 




{
  const double dt=1.0;
  
  double initial_C, initial_Cc, sum_Cp_times_p, avg_Cp, sum_p, avg_p, sum_et, avg_et, sum_qOFa, avg_qOFa, sum_qSSa, avg_qSSa, sum_qGWa, avg_qGWa; //all for determining initial concentration
  
  double a, b, y0, h, c; 
  
  int i;
  
  //set initial concentration in both boxes at long-term average
  sum_Cp_times_p=0.0; sum_p=0.0; sum_et=0.0; sum_qOFa=0.0; sum_qSSa=0.0; sum_qGWa=0.0;  //first we need the volume-weighted average concentration in precipitation
  for (i=0; i<n; i++) {
    sum_et=sum_et+et[i];
    sum_qOFa=sum_qOFa+qOFa[i];
    sum_qSSa=sum_qSSa+qSSa[i];
    sum_qGWa=sum_qGWa+qGWa[i];
    if (p[i]>0.0) {sum_Cp_times_p=sum_Cp_times_p+Cp[i]*p[i]; sum_p=sum_p+p[i];}
  }
  avg_Cp=sum_Cp_times_p/sum_p;
  avg_p=sum_p/(double)n;
  avg_et=sum_et/(double)n;
  avg_qOFa=sum_qOFa/(double)n;
  avg_qSSa=sum_qSSa/(double)n;
  avg_qGWa=sum_qGWa/(double)n;
  
  if(isotopic==true) initial_C= avg_Cp - avg_et/(avg_p - avg_qOFa) * epsilon_et; //to account for evaporative fractionation and precipitation bypassing flow
  else initial_C= avg_Cp * ((avg_p-avg_qOFa)/(avg_p - avg_qOFa - avg_et * (1-r_et))); //to account for evapoconcentration and precipitation bypassing flow
  
  initial_Cc = (avg_Cp*avg_qOFa + initial_C*(avg_qSSa+avg_qGWa))/(avg_qOFa + avg_qSSa + avg_qGWa);  //initial concentration in channel box
    
  
  ////////////////////////////////////////// UPPER BOX ///////////////////////////////////////////
  //now calculate upper box concentrations
  //the first time step is special b/c it relies on initial storage and concentration values rather than the previous time step
  
  for(i=0; i<n; i++) {
    if (isotopic==true) {  // for an isotopic tracer
      if (p[i]>0) a = (p[i] - qOFa[i])*Cp[i] - et[i]*epsilon_et ;  
      else a = - et[i]*epsilon_et ;
      b = p[i] - qOFa[i];
    } else {  //for a solute tracer
      if (p[i]>0) a = (p[i] - qOFa[i])*Cp[i]; else a = 0.0;
      b = p[i] - qOFa[i] - et[i]*(1.0-r_et);
    }
    c = p[i] - qOFa[i] - et[i] - qSSa[i] - R[i];
    if (i==0) y0=initial_C; else y0=Cu[i-1];
    if (i==0) h=su_ref; else h=su[i-1];
    
    //now we apply the formulas defined above -- this steps through each of the formulas in section S3 of the supplement
    if (b==0) { 
      Cu[i] = special_formula(a, y0, c, h, su[i]);
      Cua[i] = special_avg_formula(a, y0, c, h, su[i], dt);
    } else {
      if (fabs(c)/(p[i] + qOFa[i] + et[i] + qSSa[i] + R[i]) < 0.0001 ) {  //if we approach the exponential limit
        Cu[i] = exponential_formula(a, b, y0, h, dt);                    
        Cua[i] = exponential_avg_formula(a, b, y0, h, dt);
      }
      else {
        Cu[i] = normal_formula(a, b, y0, c, h, su[i]);
        Cua[i] = normal_avg_formula(a, b, y0, c, h, su[i], dt);
      }
    }
    
  } //next i
  
  
  
  
  ////////////////////////////////////////// LOWER BOX ///////////////////////////////////////////
  //now calculate lower box concentrations
  //the first time step is special b/c it relies on initial storage and concentration values rather than the previous time step
  
  for(i=0; i<n; i++) {
    a = R[i]*Cua[i];   //recharge takes the place of precipitation
    b = R[i];             //no ET in the lower box
    c = R[i] - qGWa[i];    //mass balance in the lower box
    if (i==0) y0=initial_C; else y0=Cl[i-1];
    if (i==0) h=sl_ref; else h=sl[i-1];
    
    //now we apply the formulas defined above
    if (fabs(c)/(R[i] + qGWa[i]) < 0.0001 ) {  //if we approach the exponential limit
      Cl[i] = exponential_formula(a, b, y0, h, dt);                    
      Cla[i] = exponential_avg_formula(a, b, y0, h, dt);
    }
    else {
      Cl[i] = normal_formula(a, b, y0, c, h, sl[i]);
      Cla[i] = normal_avg_formula(a, b, y0, c, h, sl[i], dt);
    }
    
  } //next i
  
  
  
  ////////////////////////////////////////// STREAM CHANNEL ///////////////////////////////////////////
  //now calculate stream channel concentrations
  //the first time step is special b/c it relies on initial storage and concentration values rather than the previous time step
  
  
  
  for(i=0; i<n; i++) {
    if (p[i]>0.0) a = qOFa[i]*Cp[i] + qSSa[i]*Cua[i] + qGWa[i]*Cla[i];   //input flux to channel box
         else a = qSSa[i]*Cua[i] + qGWa[i]*Cla[i];                    // if p==0 so Cp is undefined
    b = qOFa[i] + qSSa[i] + qGWa[i];             //no ET in the channel box
    c = qOFa[i] + qSSa[i] + qGWa[i] - qa[i];    //water mass balance in the channel box
    if (i==0) y0=initial_Cc; else y0=Cq[i-1];  
    if (i==0) h=sc_ref; else h=sc[i-1];
    
    //now we apply the formulas defined above  --  note that Cq equals concentration in channel box so we don't need to track them separately
    if (fabs(c)/(qOFa[i] + qSSa[i] + qGWa[i] + qa[i]) < 0.0001 ) {  //if we approach the exponential limit
      Cq[i] = exponential_formula(a, b, y0, h, dt);                    
      Cqa[i] = exponential_avg_formula(a, b, y0, h, dt);
    }
    else {
      Cq[i] = normal_formula(a, b, y0, c, h, sc[i]);
      Cqa[i] = normal_avg_formula(a, b, y0, c, h, sc[i], dt);
    }
    
  } //next i
  
  
} //end model_solute_tracers






///////////////////////////////////////////////
// here is the main routine
///////////////////////////////////////////////

// [[Rcpp::export]]
List threebox_tracers (List input)
{
  
  //now we instantiate the necessary inputs into Cpp
  
  int n =            input["n"];      //number of time steps
  NumericVector p =   input["p"];     //precipitation rate time series (average rate over time step, not cumulative depth per time step dt)
  NumericVector Cp =  input["Cp"];    //precipitation tracer concentration
  NumericVector et =  input["et"];    //potential evapotranspiration time series (average rate over time step, not cumulative depth per time step dt)
  NumericVector qOF = input["qOF"];   //overland flow bypassing upper box (at end of each time step)
  NumericVector qSS = input["qSS"];   //shallow subsurface flow bypassing lower box (at end of each time step)
  NumericVector qGW = input["qGW"];   //groundwater discharge from lower box drainage (at end of each time step)
  NumericVector qOFa = input["qOFa"];   //overland flow bypassing upper box (averaged over each time step)
  NumericVector qSSa = input["qSSa"];   //shallow subsurface flow bypassing lower box (averaged over each time step)
  NumericVector qGWa = input["qGWa"];   //groundwater discharge from lower box drainage (averaged over each time step)
  NumericVector qa =   input["qa"];   //stream discharge (averaged over each time step)
  NumericVector R =   input["R"];     //lower-box recharge (averaged over each time step)
  NumericVector su =  input["su"];    //upper box storage (at end of each time step)
  NumericVector sl =  input["sl"];    //lower box storage (at end of each time step)
  NumericVector sc =  input["sc"];    //stream channel storage (at end of each time step)
  double su_ref =    input["su_ref"];    //upper box initial storage
  double sl_ref =    input["sl_ref"];    //lower box initial storage
  double sc_ref =    input["sc_ref"];    //stream channel initial storage
  double r_et =       input["r_et"];       //relative concentration of tracer in ET flux (1=same as in source water, 0=no tracer in ET)
  double epsilon_et = input["epsilon_et"]; //isotopic fractionation of ET flux
  bool isotopic =     input["isotopic"];   //flag for whether tracer is an isotope or a solute
  
  
  
  
  //define vectors and arrays
  
  NumericVector Cq(n);  	//tracer concentration in discharge (at end of time step)
  NumericVector Cu(n);  	//tracer concentration in upper box (at end of time step)
  NumericVector Cl(n);  	//tracer concentration in lower box (at end of time step)
  NumericVector Cqa(n);  	//tracer concentration in discharge (averaged over time step)
  NumericVector Cua(n);  	//tracer concentration in upper box (averaged over time step)
  NumericVector Cla(n);  	//tracer concentration in lower box (averaged over time step)
  
  time_t start_time, end_time;
  
  time(&start_time);
  
  //and calculate tracer concentrations
  model_tracers(n, p, Cp, et, isotopic, epsilon_et, r_et, qOF, qSS, qGW, qOFa, qSSa, qGWa, R, qa, su, sl, sc, su_ref, sl_ref, sc_ref, Cq, Cu, Cl, Cqa, Cua, Cla);
  
  
  time(&end_time);
  Rprintf("\n\nTracers done %g seconds\n\n", difftime(end_time,start_time));
  
  
  //now we compile the output list
  List output;
  output["Cq"] = Cq;  //tracer concentration in discharge (instantaneous value at end of time step)
  output["Cu"] = Cu;	//tracer concentration in upper box (instantaneous value at end of time step)
  output["Cl"] = Cl;	//tracer concentration in lower box (instantaneous value at end of time step)
  output["Cqa"] = Cqa;  //tracer concentration in discharge (average over time step)
  output["Cua"] = Cua;	//tracer concentration in upper box (average over time step)
  output["Cla"] = Cla;	//tracer concentration in lower box (average over time step)
  
  return output;  //and return the output list
  
  
} //end twobox_solute_tracers




