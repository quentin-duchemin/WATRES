
// age-tracking transit time distributions for
// three-box conceptual hydrological model 
// for benchmark testing of ensemble hydrograph separation


// this version is designed to be called from R using Rcpp...
// NOT for stand-alone use!
  
  
  // version 1.0  Build 2020.08.30
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



// this is the formula for integrating TTDs forward through time
inline double integral( double &a, double &b, double &y0, double &c, double &h, double dt ) 
{ 
  if (b==0.0) return( y0 - (a/c)*log(h/(h+c*dt)) );   // logarithmic formula if b==0
  else if ( fabs(c) < 0.0001 * fabs(b) ) return ( (a/b) + (y0 - (a/b))*exp(-b*dt/h) );  // exponential formula if c is nearly zero
  else return( (a/b) + (y0 - (a/b))*pow(h/(h+c*dt), (b/c)) );   // the normal formula
}



// this is the formula for averaging TTDs over each time step
inline double avg_integral( double &a, double &b, double &y0, double &c, double &h, double dt ) 
{ 
  if (b==0.0) return( y0 - (a/c)*(1 + ((h+c*dt)/(c*dt))*log(h/(h+c*dt))) );  // logarithmic formula if b==0
  else if ( fabs(c) < 0.0001 * fabs(b) ) return ( (a/b) + (y0 - (a/b))*(h/(b*dt))*(1.0 - exp(-b*dt/h)) );  // exponential formula if c is nearly zero
  else return( (a/b) + (y0 - (a/b))*(h/(b-c)/dt)*(1.0 - pow(h/(h+c*dt), (b/c - 1.0))) );   // the normal formula
}




void model_age_tracking(int nstart, int nstartsave, int n, 
                        double su_ref, double sl_ref, double sc_ref,
                        int nttd,
                        NumericVector p, NumericVector et, 
                        NumericVector qOF, NumericVector qSS, NumericVector qGW, NumericVector q, 
                        NumericVector qOFa, NumericVector qSSa, NumericVector qGWa, NumericVector qa, NumericVector R, 
                        NumericVector su, NumericVector sl, NumericVector sc,
                        NumericMatrix &TTD_Su, NumericMatrix &TTD_Sl, 
                        NumericMatrix &TTD_D, NumericMatrix &TTD_GWa, 
                        NumericMatrix &TTD,  NumericMatrix &rank_storage
)
// calculates TTD's by age tracking in the three-box model
  
  //receives as input:
  //	p: precipitation time series of length n
  //	su_ref, sl_ref, sc_ref: reference volumes for upper, lower, and channel boxes
  //		these will be the equilibrium volumes at p=p_bar
  //	dt: time step
  //	nttd: number of ttd time steps
  //  qOF, qSS, qGW, q: discharge from overland flow, shallow subsurface flow, groundwater flow, and total streamflow (instantaneous value at end of time step)
  //  qOFa, qSSa, qGWa, qa, R: overland flow, shallow subsurface flow, groundwater flow, total streamflow, and groundwater recharge (averages over each time step)
  //  su, sl, sc: storage time series (instantaneous s at end of each time interval) for upper, lower, and channel boxes
  
  
  //returns as output:
  //  TTD_Su, TTD_Sl: transit time distributions in upper and lower boxes
  //  TTD_D, TTD_GWa: transit time distributions in upper box drainage and lower box groundwater flow
  //  TTD, TTDa: transit time distribution in streamflow (instantaneous and averaged over each time interval)
  //	fwd_TTD, fwd_TTDa: forward transit time distribution in streamflow (instantaneous and averaged over each time interval)
  
  
  
{
  const double dt = 1.0;
  
  double p_bar; //average precipitation rate
  double qOF_bar; //average bypassing flow
  double q_bar;   //average discharge
  double et_bar; //average actual evapotranspiration rate
  double qSS_bar; //average subsurface stormflow
  double qGW_bar; //average groundwater flow
  
  double initial_TTD_su[nttd];  //these will hold the initial (equilibrium) ttd's for the upper, lower, and channel boxes
  double initial_TTD_sl[nttd];
  double initial_TTD_sc[nttd];
  double initial_TTD_D[nttd];
  double initial_TTD_GW[nttd];
  double initial_TTD_I[nttd];
  
  int i, k;
  
  double a, b, y0, h, c;
  
  double cTTD[nttd];
  double rSTO[nttd];
  double sT_u;
  double sT_l;
  double sT_c;
  double TTD_GWa_old;
  double TTD_Dold;
  double TTD_old[nttd];
  double TTD_Slold[nttd];
  double TTD_Suold[nttd];
  
  
  //first calculate average fluxes for the whole time series (we will need these to set initial TTDs)
  p_bar=0.0; for (i=0; i<n; i++) p_bar = p_bar+p[i]; p_bar = p_bar / (double) n;
  et_bar=0.0; for (i=0; i<n; i++) et_bar = et_bar+et[i]; et_bar = et_bar / (double) n;
  qOF_bar=0.0; for (i=0; i<n; i++) qOF_bar = qOF_bar+qOFa[i]; qOF_bar = qOF_bar / (double) n;
  qSS_bar=0.0; for (i=0; i<n; i++) qSS_bar = qSS_bar+qSSa[i]; qSS_bar = qSS_bar / (double) n;
  qGW_bar=0.0; for (i=0; i<n; i++) qGW_bar = qGW_bar+qGWa[i]; qGW_bar = qGW_bar / (double) n;
  q_bar=0.0; for (i=0; i<n; i++) q_bar = q_bar+qa[i]; q_bar = q_bar / (double) n;
  
  
  
  
  ////////////////////    UPPER BOX    ///////////////////////////////////////
    //calculate ttd for upper box (fraction of upper box storage that is composed of same-time-step precip (lag0) and precip from 1, 2, 3... time steps ago
                                   //first calculate the intial ttd, assuming a steady-state equilibrium with p_bar and su_ref
                                   //note that because we assume that both drainage and ET are not age-selective, it doesn't matter how much water leaves by each pathway
  
  // note that "initial" means "before time step 0"... since time step zero is the first time step with precipitation inputs
  // these initial values are at equilibrium with a constant average input of (p_bar-q0_bar) and constant storage of su_ref
  TTD_Su(0,0) = (1.0 - exp(-dt*(p_bar-qOF_bar)/su_ref));
  initial_TTD_D[0] = 1.0 - (su_ref/((p_bar-qOF_bar)*dt))*(1.0 - exp(-(p_bar-qOF_bar)*dt/su_ref));  //we will need this one for the lower box
  for (k=1; k<nttd; k++) {
    TTD_Su(0,k) = TTD_Su(0,k-1) * exp(-(p_bar-qOF_bar)*dt/su_ref); 
    initial_TTD_D[k] = TTD_Su(0,k-1) * (su_ref/((p_bar-qOF_bar)*dt))*(1.0 - exp(-(p_bar-qOF_bar)*dt/su_ref));
  }
  
  
  // LOWER
  TTD_Sl(0,0) = initial_TTD_D[0]  * (1.0 - exp(-qGW_bar*dt/sl_ref));
  initial_TTD_GW[0] = initial_TTD_D[0]  * (1.0 - sl_ref/(qGW_bar * dt) * (1.0 - exp(-qGW_bar*dt/sl_ref)));
  for (k=1; k<nttd; k++) {
    TTD_Sl(0,k)=  initial_TTD_D[0]  + (TTD_Sl(0,k-1)-  initial_TTD_D[k] ) * exp(-qGW_bar*dt/sl_ref);
    initial_TTD_GW[k] = initial_TTD_D[k] + (TTD_Sl(0,k-1)- initial_TTD_D[k]) * (sl_ref/(qGW_bar * dt)) * (1.0 - exp(-qGW_bar*dt/sl_ref));   // we need this for initializing the channel box
  }
  
  
  // CHANNEL
  //first calculate the intial ttd, assuming a steady-state equilibrium  at constant input rates of q_OFbar, q_SSbar, and q_GWbar and storage of sc_ref
  initial_TTD_I[0] = (qOF_bar + qSS_bar*initial_TTD_D[0] + qGW_bar*initial_TTD_GW[0])/(qOF_bar + qSS_bar + qGW_bar);
  TTD(0,0) = initial_TTD_I[0]  * (1.0 - exp(-q_bar*dt/sc_ref));
  for (k=1; k<nttd; k++) {
    initial_TTD_I[k] = (qSS_bar*initial_TTD_D[k] + qGW_bar*initial_TTD_GW[k])/(qOF_bar + qSS_bar + qGW_bar);
    TTD(0,k) = initial_TTD_I[k] + (TTD(0,k-1) - initial_TTD_I[k]) * exp(-q_bar*dt/sc_ref);
  }
  
  
  // now walk through array
  for (i=nstart; i<n; i++) {
    if (i%1000==0){    std::cout << i << " ";}
    for (k=0; k<nttd; k++) {
      
      
      // UPPER
      b = p[i] - qOFa[i];
      if (k==0) a = b; else a = 0.0;
      if (k==0) y0 = 0.0;
      else {
       y0 = TTD_Suold[k-1];
      }
      if (i==nstart) h = su_ref; else h = su[i-1];
      c = p[i] - qOFa[i] - qSSa[i] - R[i];

        TTD_Su(0,k) = integral( a, b, y0, c, h, dt );
        TTD_Dold = TTD_D(0,k);
        TTD_D(0,k) = avg_integral( a, b, y0, c, h, dt );

      
      // LOWER
      b = R[i];
      a = R[i] * TTD_Dold; 
      if (k==0) y0 = 0.0;
      else {
        y0 = TTD_Slold[k-1];
      }
      if (i==nstart) h = sl_ref; else h = sl[i-1];
      c = R[i] - qGWa[i];
        TTD_Sl(0,k) = integral( a, b, y0, c, h, dt );
        TTD_GWa_old = TTD_GWa(0,k);
        TTD_GWa(0,k) = avg_integral( a, b, y0, c, h, dt );
    
      
    // CHANNEL
      b = qOFa[i]+qSSa[i]+qGWa[i];
      if (k==0){
          a = qOFa[i]+qSSa[i]*TTD_Dold+qGWa[i]*TTD_GWa_old;
      } 
      else{
          a = qSSa[i]*TTD_Dold+qGWa[i]*TTD_GWa_old;
      } 
      if (k==0) y0 = 0.0;
      else {
        if (i<=nstartsave) y0 = TTD_old[k-1]; else  y0 = TTD(i-1-nstartsave, k-1) ; 
      }
      if (i==nstart) h = sc_ref; else h = sc[i-1];
      c = qOFa[i]+qSSa[i]+qGWa[i]-qa[i];
      
      if (i<=nstartsave){
        TTD(0,k) = integral( a, b, y0, c, h, dt );
        if (k==0) {
          cTTD[k] = TTD(0,k);
          
          sT_u = su[i] * TTD_Su(0,k);
          sT_l = sl[i] * TTD_Sl(0,k);
          sT_c = sc[i] * TTD(0,k);
          rSTO[k] = sT_u + sT_l + sT_c;
          } 
        else {
          cTTD[k]=cTTD[k-1]+TTD(0,k);
          sT_u = su[i] * TTD_Su(0,k);
          sT_l = sl[i] * TTD_Sl(0,k);
          sT_c = sc[i] * TTD(0,k);
          rSTO[k] = rSTO[k-1] + sT_u + sT_l + sT_c;
          }
      }
      else{
        TTD(i-nstartsave,k) = integral( a, b, y0, c, h, dt );
        if (k==0) {
          cTTD[k] = TTD(i-nstartsave,k);
          sT_u = su[i] * TTD_Su(0,k);
          sT_l = sl[i] * TTD_Sl(0,k);
          sT_c = sc[i] * TTD(i-nstartsave,k);
          rSTO[k] = sT_u + sT_l + sT_c;
          rank_storage(i-nstartsave,k) = rSTO[k];
        } 
        else {
          cTTD[k]=cTTD[k-1]+TTD(i-nstartsave,k);
          sT_u = su[i] * TTD_Su(0,k);
          sT_l = sl[i] * TTD_Sl(0,k);
          sT_c = sc[i] * TTD(i-nstartsave,k);
          rSTO[k] = rSTO[k-1] + sT_u + sT_l + sT_c;
          rank_storage(i-nstartsave,k) = rSTO[k];
        }
      }
    }
    for (k=0; k<nttd; k++){
      TTD_old[k] = TTD(0,k);
      TTD_Slold[k] = TTD_Sl(0,k);
      TTD_Suold[k] = TTD_Su(0,k);
    }

    
    
  }
 } //end model_age_tracking


void cumulativeSum(NumericMatrix &arr, double (&cTTD)[], int nttd){
  cTTD[1] = arr(0,1);
  for(int i = 2; i < nttd ; i++)
    cTTD[i] = cTTD[i-1]+ arr(0,i);
}








///////////////////////////////////////////////
  // here is the main routine
///////////////////////////////////////////////
  
  // [[Rcpp::export]]
List threebox_age_tracking (List input)
{  
  
  //now we instantiate the inputs into Cpp
  
  int n =             input["n"];   //number of time steps
  int nstart =        input["nstart"];
  int nstartsave =    input["nstartsave"];
  NumericVector p =   input["p"];   //precipitation rate time series (average rate over time step, not cumulative depth per time step dt)
  NumericVector et =  input["et"];  //actual evapotranspiration time series (average rate over time step, not cumulative depth per time step dt)
  NumericVector qOF = input["qOF"]; //overland flow discharge that bypasses upper box (at end of time step)
  NumericVector qSS = input["qSS"]; //shallow subsurface flow that bypasses lower box (at end of time step)
  NumericVector qGW = input["qGW"]; //groundwater flow draining from lower box (at end of time step)
  NumericVector q =   input["q"];   //total discharge (at end of time step)
  NumericVector qOFa = input["qOFa"]; //overland flow discharge that bypasses upper box (average rate over time step)
  NumericVector qSSa = input["qSSa"]; //shallow subsurface flow that bypasses lower box (average rate over time step)
  NumericVector qGWa = input["qGWa"]; //groundwater flow draining from lower box (average rate over time step)
  NumericVector qa =   input["qa"];   //total discharge (average rate over time step)
  NumericVector R =   input["R"];   //recharge to lower box (average rate over time step)
  NumericVector su =  input["su"];  //upper box storage (instantaneous value at end of time step)
  NumericVector sl =  input["sl"];  //lower box storage (instantaneous value at end of time step)
  NumericVector sc =  input["sc"];  //stream channel storage (instantaneous value at end of time step)
  
  double su_ref =     input["su_ref"];  //upper box reference volume
  double sl_ref =     input["sl_ref"];  //lower box reference volume
  double sc_ref =     input["sc_ref"];  //stream channel reference volume
  int nttd =          input["nttd"];    //length of ttd (in number of time steps)
  
  
  
  //define vectors and arrays
  
  NumericMatrix TTD(n-nstartsave+1,nttd);      //transit time distribution (instantaneous at end of time step)
  NumericMatrix fwd_TTD(1,nttd);	//forward transit time distribution (instantaneous at end of time step)
  NumericMatrix TTDa(1,nttd);      //transit time distribution (averaged over time step)
  NumericMatrix TTD_GWa(1,nttd);   //transit time distribution of drainage from lower box (averaged over time step)
  NumericMatrix TTD_SSa(1,nttd);	  //transit time distribution of drainage from upper box (averaged over time step)
  NumericMatrix TTD_Su(1,nttd);   //transit time distribution of upper box (instantaneous at end of time step)
  NumericMatrix TTD_Sl(1,nttd);	  //transit time distribution of lower box (instantaneous at end of time step)
  
  NumericMatrix rank_storage(n-nstartsave+1,nttd);
  
  
  time_t start_time, end_time;
  
  time(&start_time);
  
  
  //and calculate ttd's
                                   model_age_tracking(nstart, nstartsave, n, su_ref, sl_ref, sc_ref, nttd, p, et, qOF, qSS, qGW, q, qOFa, qSSa, qGWa, qa, R, su, sl, sc, TTD_Su, TTD_Sl, TTD_SSa, TTD_GWa, TTD, rank_storage);
                                   
                                   time(&end_time);
                                   Rprintf("\n\nTTD's done %g seconds\n\n", difftime(end_time,start_time));
                                   
                                   // 
                                     // int nttd_ywf = 7202;
                                   // 
                                     // 
                                     // NumericMatrix TTD_ywf(1,nttd_ywf);      //transit time distribution (instantaneous at end of time step)
                                   // NumericMatrix fwd_TTD_ywf(1,nttd_ywf);	//forward transit time distribution (instantaneous at end of time step)
                                   // NumericMatrix TTDa_ywf(1,nttd_ywf);      //transit time distribution (averaged over time step)
                                   // NumericMatrix fwd_TTDa_ywf(1,nttd_ywf);	//forward transit time distribution (averaged over time step)
                                   // NumericMatrix TTD_GWa_ywf(1,nttd_ywf);   //transit time distribution of drainage from lower box (averaged over time step)
                                   // NumericMatrix TTD_SSa_ywf(1,nttd_ywf);	  //transit time distribution of drainage from upper box (averaged over time step)
                                   // NumericMatrix TTD_Su_ywf(1,nttd_ywf);   //transit time distribution of upper box (instantaneous at end of time step)
                                   // NumericMatrix TTD_Sl_ywf(1,nttd_ywf);	  //transit time distribution of lower box (instantaneous at end of time step)
                                   // 
                                     
                                     //  model_age_tracking_young_water_fraction(nstart, n, su_ref, sl_ref, sc_ref, nttd_ywf, p, et, qOF, qSS, qGW, q, qOFa, qSSa, qGWa, qa, R, su, sl, sc, TTD_Su_ywf, TTD_Sl_ywf, TTD_SSa_ywf, TTD_GWa_ywf, TTD_ywf, young_water_fraction, TTD);
                                   
                                   
                                   
                                   
                                   
                                   //now we compile the output list
                                   List output;
                                   output["TTD"] = TTD;         //transit time distribution of streamflow (instantaneous at end of time step)

                                   
                                   
                                   output["rank_storage"]=rank_storage;
                                   
                                   
                                   return output;  //and return the output list
                                   
                                   
                                   } //end twobox age tracking




