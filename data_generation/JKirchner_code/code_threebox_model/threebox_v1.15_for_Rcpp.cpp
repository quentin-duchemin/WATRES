
// three-box conceptual hydrological model 
// for benchmark testing of ensemble rainfall-runoff analysis


// this version is designed to be called from R using Rcpp...
// NOT for stand-alone use!


// version 1.15  Build 2021.02.03
// Author: James Kirchner, ETH Zurich
//
// Copyright 2021 ETH Zurich and James Kirchner
// Public use allowed under GNU Public License v. 3 (see http://www.gnu.org/licenses/gpl-3.0.en.html or the attached license.txt file)
//
// READ THIS CAREFULLY:
// ETH Zurich and James Kirchner make ABSOLUTELY NO WARRANTIES OF ANY KIND, including NO WARRANTIES, expressed or implied, that this software is 
//    free of errors or is suitable for any particular purpose.  Users are solely responsible for determining the suitability and 
//    reliability of this software for their own purposes.


// The calculations here follow the general scheme outlined in
// Kirchner, J.W., Quantifying catchment response to precipitation, in models and field data, 
// using effect tracking and ensemble rainfall-runoff analysis (HESS, in review).  


// The functions CKRK45 and integr are the core of the Cash-Karp 4/5th order Runge-Kutta integration scheme.
// These functions are adapted from the calculation scheme (but not the code) of Press, Teukolsky, Vetterling, and Flannery, 
// Numerical Recipes in C, 2nd Edition, Cambridge University Press, 1992.


// To run this script, users will need to install the Rcpp library.
// Windows users will also need to install Rtools; see https://cran.r-project.org/bin/windows/Rtools/


// changes in version 1.15 relative to version 1.0:
//   aggregates time steps according to the parameter agg
//   calculates RRD by perturbing precip in agg consecutive time steps to zero



#include <Rcpp.h>
using namespace Rcpp;

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define MAXSTEPS 10000
#define TINY 1.0e-8    //square root of nominal 16-digit machine precision ( <==>53-bit mantissa of R numerics )
//staying within this relative tolerance should keep us clear of problematic roundoff errors


///////////////////////////////////////////
// ds/dt functions for RK4/5 integration 
// These are done as macro calls rather than as functions, for speed.  They *require* that the paramset inside the calling function is called "par".

// this is the normal one with storage-dependent bypassing
#define dsdt_normal(s) ( par.p*(1-pow(s/(s+par.s_ref), par.a)) - par.D_ref * pow(s/par.s_ref, par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) )

// this one is special for negative values of s (it's antisymmetrical with respect to s, so that behavior is reasonable for s<0 and we don't throw an error with noninteger b...)
#define dsdt_neg(s) ( par.p + par.D_ref * pow(fabs(s/par.s_ref), par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) )

// this one is special for cases where we have turned storage-dependent bypassing off and use fixed fraction instead
#define dsdt_nobypass(s) ( par.p*(1.0-par.c) - par.D_ref * pow(s/par.s_ref, par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) )

// this one is special for cases where we have turned storage-dependent bypassing off and use fixed fraction instead
#define dsdt_neg_nobypass(s) ( par.p*(1.0-par.c) + par.D_ref * pow(fabs(s/par.s_ref), par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) )






// this structure is used to pass parameters for the Runge-Kutta routine (for both upper and lower boxes)
struct paramset{ 
  double p;      
  double pet;
  double s_wp;
  double s_o;
  double s_ref;
  double D_ref;
  double b;
  double a;
  double c;
};



/*   This is the function that was replaced by the macros above.
 // Here is the mass balance equation of the nonlinear reservor
 // (which determines the rate of change of storage for the Cash-Karp RK procedure)
 double dsdt(paramset &par,  //these are passed by reference for sake of efficiency
 double &s)
 
 // receives as input
 // par:      parameter set
 // s:        storage value at which derivative is required
 
 //returns as output
 //derivative at specified storage value
 
 { 
 // make the leakage function antisymmetrical with respect to s, so that behavior is reasonable for s<0 and we don't throw an error with noninteger b...
 if (s<=0.0) return(  par.p + par.D_ref * pow(fabs(s/par.s_ref), par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) );  
 else if (par.a==-1) return(  par.p - par.D_ref * pow(s/par.s_ref, par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) );  //no bypassing flow
 else return(  par.p*(1-pow(s/(s+par.s_ref), par.a)) - par.D_ref * pow(s/par.s_ref, par.b) - par.pet * fmin(fmax((s-par.s_wp)/(par.s_o-par.s_wp), 0.0), 1.0) );  //with bypassing flow
 } //end dsdt
 */



// This is the procedure that takes one Cash-Karp Runge-Kutta step
void CKRK45(paramset &par,
            double &s0,
            double &h,
            double &s,
            double &err)
  
  //takes one step of length h using Cash-Karp embedded 4th-5th order Runge-Kutta integration
  //this is simpler than a standard Cash-Karp routine because this is a single ODE rather than a system of them,
  //and it is a function only of storage and not time
  
  //receives as input (these are all passed by reference to save time, but are not altered here)
  //par: parameter set to be passed to derivative function
  //s0:  initial value of s
  //h:   step length
  
  //returns with new values for
  //s:   value of s at end of step
  //err: Cash-Karp error estimate
  
{  static double // keep Cash-Karp coefficients as static to avoid re-initializing
  b21=0.2, b31=3.0/40.0, b32=9.0/40.0, b41=0.3, b42=-0.9, b43=1.2,
    b51=-11.0/54.0, b52=2.5, b53=-70.0/27.0, b54=35.0/27.0,
    b61=1631.0/55296.0, b62=175.0/512.0, b63=575.0/13824.0,
    b64=44275.0/110592.0, b65=253.0/4096.0,
    c1=37.0/378.0, c3=250.0/621.0, c4=125.0/594.0, c6=512.0/1771.0, 
    dc1=37.0/378.0-2825.0/27648.0, dc3=250.0/621.0-18575.0/48384.0,
    dc4=125.0/594.0-13525.0/55296.0, dc5=-277.00/14336.0, dc6=512.0/1771.0-0.25;
  
  double k1, k2, k3, k4, k5, k6, s_temp;  //these will hold the derivative evaluations
  
  
  if (par.a==-1) {  //if we are not using storage-dependent bypassing
    
    if (s0<=0.0) k1 = dsdt_neg_nobypass(s0); //step 1
    else k1 = dsdt_nobypass(s0);  
    
    s_temp = s0 + h*b21*k1;  //step 2
    if (s_temp<=0.0) k2 = dsdt_neg_nobypass(s_temp);
    else k2 = dsdt_nobypass(s_temp);
    
    s_temp = s0 + h*(b31*k1 + b32*k2);  //step 3
    if (s_temp<=0.0) k3 = dsdt_neg_nobypass(s_temp);
    else k3 = dsdt_nobypass(s_temp);
    
    s_temp = s0 + h*(b41*k1 + b42*k2 + b43*k3);  //step 4
    if (s_temp<=0.0) k4 = dsdt_neg_nobypass(s_temp);
    else k4 = dsdt_nobypass(s_temp);
    
    s_temp = s0 + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4);  //step 5
    if (s_temp<=0.0) k5 = dsdt_neg_nobypass(s_temp);
    else k5 = dsdt_nobypass(s_temp);
    
    s_temp = s0 + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5);  //step 6
    if (s_temp<=0.0) k6 = dsdt_neg_nobypass(s_temp);
    else k6 = dsdt_nobypass(s_temp);
    
  }  else {   //if we *are* using storage-dependent bypassing
    
    if (s0<=0.0) k1 = dsdt_neg(s0); //step 1
    else k1 = dsdt_normal(s0);
    
    s_temp = s0 + h*b21*k1;  //step 2
    if (s_temp<=0.0) k2 = dsdt_neg(s_temp);
    else k2 = dsdt_normal(s_temp);
    
    s_temp = s0 + h*(b31*k1 + b32*k2);  //step 3
    if (s_temp<=0.0) k3 = dsdt_neg(s_temp);
    else k3 = dsdt_normal(s_temp);
    
    s_temp = s0 + h*(b41*k1 + b42*k2 + b43*k3);  //step 4
    if (s_temp<=0.0) k4 = dsdt_neg(s_temp);
    else k4 = dsdt_normal(s_temp);
    
    s_temp = s0 + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4);  //step 5
    if (s_temp<=0.0) k5 = dsdt_neg(s_temp);
    else k5 = dsdt_normal(s_temp);
    
    s_temp = s0 + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5);  //step 6
    if (s_temp<=0.0) k6 = dsdt_neg(s_temp);
    else k6 = dsdt_normal(s_temp);
    
  }  // end if
  
  
  //here is the 5th-order estimate for the next value of s
  s = s0 + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6);
  
  //here is the Cash-Karp error estimate (difference between 5th-order and 4th-order estimates, here as absolute value)
  err = fabs(h*(dc1*k1 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6));
  
} //end CKRK45




// This is the ODE integration driver
void integr(paramset par,
            double s0,
            double delta_t,
            double tol,
            double &s)
  // Integrates storage equation forward through one time step of length delta_t, using using Cash-Karp embedded 4th/5th order 
  // Runge-Kutta integration with adaptive stepsize control to keep error in storage below specified tolerance
  
  //receives as input
  //par:       parameter set to be passed through to Cash-Karp RK routine and then to derivative function
  //s0:        initial value of s
  //delta_t:   step length
  //tol:       error tolerance for s (absolute, not relative)
  
  //returns with new value for
  //s:         value of s at end of step
  
{
  double t,  //current time -- runs from 0 to delta_t within every time step
  h,  //step length
  last_s,
  err,
  next_h;  //another step length
  
  int iter;  //iteration counter
  
  tol = fmax(tol, s0*TINY); //don't let tol be a smaller fraction of s0 than TINY, to avoid going below roundoff error
  next_h = delta_t; //start by trying step equal to whole delta-t
  t = 0.0;     //initialize time
  iter = 0;   //initialize iteration counter
  last_s = s0;  //initial S
  
  do{
    do {
      h = next_h; 
      CKRK45(par, last_s, h, s, err);  //try a CKRK step of length h, integrating from last_s to s
      if (s<=0.0) {next_h = h/5.0; err=2.0*tol;}      //special for this particular application: if s is negative, cut step size and set err>tol so that we will iterate again... 
      else next_h = 0.9 * fmax(h * pow(tol/err, 0.25), h/5.0) ;    //update stepsize according to Cash-Karp approach (this will only be used if err>tol and we have to try again)
      iter++; //update iteration counter
      if(iter>(MAXSTEPS-10)) {REprintf("iter %i  t %g   s %g  s-last_s%g  h %g  err %g \n", iter, t, s, (s-last_s), h, err);  //print diagnostic information if we're about to crash through MAXSTEPS limit
        if(iter>MAXSTEPS) Rcpp::stop("sorry, iteration limit exceeded in intgr!"); } //exit if we have exceeded the iteration limit
    } while (isnan(err) || (err>tol));  //until we are within allowed error tolerance (and thus CKRK step was successful).  Note that we need to trap err values of NaN, otherwise they get through!
    
    //if CKRK step was successful, we get to this point.  Now we...
    last_s = s;  //update last_s to current s (for the next step, assuming there will be one)
    t = t + h;  //update current time
    next_h = 0.9 * fmin(h * pow(tol/err, 0.2), h*5.0);     //update stepsize according to C-K approach
    next_h = fmin(next_h, delta_t-t);  //update stepsize, but don't step beyond the end of delta_t
  } while (t<delta_t);  //repeat until we are at end of required time step
  
  return;  //normal termination
  
} //end integr




//this is the three-box model itself
void threebox(NumericVector p, 
              NumericVector pet,
              int n, 
              double bu, double bl, double bc,
              double aOF, double aSS,
              double su_ref, double sl_ref, double sc_ref,
              double su_init, double sl_init, double sc_init, 
              double s_wp, double s_o, double D_ref, double qGW_ref, double q_ref,
              double tol,
              double dt,
              NumericVector &qOF, NumericVector &qSS, NumericVector &qGW, NumericVector &q, 
              NumericVector &qOFa, NumericVector &qSSa, NumericVector &qGWa, NumericVector &qa, 
              NumericVector &et, NumericVector &R, NumericVector &su, NumericVector &sl, NumericVector &sc, NumericVector &D, NumericVector &Da)
  //calculates water fluxes in the three-box model
  
  
  
  //receives as input:
  //	p:                precipitation time series of length n
  //  pet:              potential ET time series of length n
  //  n:                number of time steps
  //  bu, bl, bc:       exponents for upper, lower, and channel box drainage equations
  //  aOF, aSS:         exponents for upper and lower box input partitioning  
  //	su_ref, sl_ref, sc_ref:   reference volumes for upper, lower, and channel boxes
  //	                   	these will be the equilibrium volumes at average P and PET
  //	su_init, sl_init, sc_init: initial storages for upper, lower, and channel boxes
  //                        these will differ from su_ref, sl_ref, and sc_ref if we are doing perturbations (and thus not are starting at time zero)
  //  s_wp, s_o, D_ref, qGW_ref: parameters derived from initialization routine (equilibrium with average P and PET)
  //	tol:               Runge-Kutta solution tolerance
  
  //returns as output:
  //  qOF:       overland flow discharge bypassing upper box (at end of each time step)
  //  qSS:       shallow subsurface flow bypassing lower box (at end of each time step)
  //  qGW:       groundwater flow draining from lower box (at end of each time step)
  //  q:         discharge time series (qOF+qSS+qGW) processed through channel storage
  //  qOFa:      overland flow discharge bypassing upper box (averaged over time step)
  //  qSSa:      shallow subsurface flow bypassing lower box (averaged over time step)
  //  qGWa:      groundwater flow draining from lower box (averaged over each time step)
  //  qa:        discharge time series (qOF+qSS+qGW processed through channel storage, averaged over each time step)
  //  et:        evapotranspiration time series (average ET over each time interval)
  //  R:         recharge to lower box (average over each time step)
  //  su:        upper box storage (instantaneous s at end of each time interval)
  //  sl:        lower box storage (instantaneous s at end of each time interval)
  //  sc:        channel box storage (instantaneous s at end of each time interval)
  
  
{	
  int i;
  paramset par;
  
  double eta_OF;    // overland flow partitioning coefficient (at end of time step)
  double eta_OFa;   // overland flow partitioning coefficient (averaged over time step)
  double eta_SS;    // shallow subsurface flow partitioning coefficient (at end of time step)
  double eta_SSa;   // shallow subsurface flow partitioning coefficient (averaged over time step)
  double su_pre;
  double sl_pre;
  double sc_pre;
  
  //  NumericVector D(n);         // drainage from upper box (at end of time step)
  //  NumericVector Da(n);        // drainage from upper box (averaged over time step)
  
  
  
  //////////////////////////    UPPER BOX    /////////////////////////////////////
  //run model for upper box
  
  //set parameters for upper box
  par.s_wp = s_wp;    //wilting point storage
  par.s_o = s_o;        //storage at which ET is independent of s
  par.D_ref = D_ref;  //reference drainage rate
  par.s_ref = su_ref; //reference storage level
  par.b = bu;         //drainage exponent
  if (aOF<=0) {         //negative values are a flag to indicate that we should be using a fixed overland flow fraction rather than a storage-dependent one
    par.a = -1;
    par.c = - aOF;      //here is the fixed overland flow fraction  
  } else {
    par.a = aOF;        //lower box partitioning exponent (for storage-dependent overland flow)
    par.c = 0.0;
  }
  
  
  //now step through the time series
  for (i=0; i<n; i++) {
    par.p = p[i];   par.pet = pet[i];      //copy over precipitation and PET
    if (i==0) su_pre = su_init; else su_pre = su[i-1];   //need to do start with su_init at first time step
    integr(par, su_pre, dt, tol, su[i]);  //integrate for new storage value
    et[i] = pet[i] * 0.5 * (fmin(fmax((su_pre-s_wp)/(s_o-s_wp), 0.0), 1.0) + fmin(fmax((su[i]-s_wp)/(s_o-s_wp), 0.0), 1.0) ); //et calculated as average of beginning and ending values
    if (par.a == -1) {
      eta_OF = par.c;   //overland flow fraction of precipitation (fixed)
      eta_OFa = par.c;   //overland flow fraction of precipitation (average over time step - also fixed)
    } else {
      eta_OF = pow(su[i]/(su[i]+su_ref), aOF);   //bypassing fraction of precipitation (at end of time step)
      eta_OFa = 0.5*( pow(su_pre/(su_pre+su_ref), aOF) + eta_OF );   //bypassing fraction of precipitation (averaged over time step)
    }
    D[i] = D_ref * pow((su[i]/su_ref), bu); //drainage rate at end of time step
    Da[i] = p[i]*(1.0-eta_OFa) - et[i] - (su[i]-su_pre)/dt; //average leakage from upper box over time interval, calculated from mass balance
    qOF[i] = eta_OF * p[i];  //calculate instantaneous overland flow (bypassing upper box at end of time step)
    qOFa[i] = eta_OFa * p[i];  //calculate time-averaged overland flow (bypassing upper box averaged over time step)
  } //next i
  
  
  
  //////////////////////////    LOWER BOX    /////////////////////////////////////
  //run model for lower box
  //transfer parameters for lower box (ET parameters will have no effect here, because pet=0 for lower box)
  
  par.pet = 0.0;      //no ET from lower box!                      //note naming is confusing here because we are recycling the parameter set that we used for the upper box
  par.s_wp = 0.0;     //wilting point storage (has no effect)
  par.s_o = 1.0;       //also has no effect
  par.D_ref = qGW_ref; //reference drainage rate
  par.s_ref = sl_ref; //reference storage level
  par.b = bl;         //drainage exponent
  
  if (aSS<=0) {         //negative values are a flag to indicate that we should be using a fixed subsurface flow fraction rather than a storage-dependent one
    par.a = -1;
    par.c = - aSS;      //here is the fixed subsurface flow fraction  
  } else {
    par.a = aSS;        //lower box partitioning exponent (for storage-dependent subsurface flow)
    par.c = 0.0;
  }
  
  //for lower box, pet=0 and p=D, representing drainage from upper box instead of precipitation
  
  
  //now integrate the lower box
  for (i=0; i<n; i++) {
    par.p = Da[i];
    if (i==0) sl_pre = sl_init; else sl_pre = sl[i-1];   //need to do start with sl_init at first time step
    integr(par, sl_pre, dt, tol, sl[i]);  //integrate for new storage value
    if (par.a == -1) {
      eta_SS = par.c;   //shallow subsurface flow fraction of upper box drainage (fixed)
      eta_SSa = par.c;   //shallow subsurface flow fraction of upper box drainage (average over time step - also fixed)
    } else {
      eta_SS = pow(sl[i]/(sl[i]+sl_ref), aSS);   //shallow subsurface flow fraction bypassing lower box (at end of time step)
      eta_SSa = 0.5*( eta_SS + pow(sl_pre/(sl_pre+sl_ref), aSS) );   //shallow subsurface flow fraction bypassing lower box (averaged over time step)
    }
    qSS[i] = eta_SS * D[i];  //instantaneous shallow subsurface flow (bypassing lower box at end of time step)
    qSSa[i] = eta_SSa * Da[i];  //time-averaged shallow subsurface flow (bypassing lower box averaged over time step)
    R[i] = Da[i]*(1.0-eta_SSa); //recharge to lower box (averaged over time interval) 
    qGW[i] = qGW_ref * pow((sl[i]/sl_ref), bl); //gw discharge at end of time interval
    qGWa[i] = R[i] - (sl[i]-sl_pre)/dt;  //average gw discharge over time interval, calculated from mass balance
  }
  
  
  
  //////////////////////////    CHANNEL BOX    /////////////////////////////////////
  //run model for channel box
  //transfer parameters for channel box (ET parameters will have no effect here, because pet=0 for lower box)
  
  par.pet = 0.0;      //no ET from channel box!                      //note naming is confusing here because we are recycling the parameter set that we used for the upper box
  par.s_wp = 0.0;     //wilting point storage (has no effect)
  par.s_o = 1.0;      //also has no effect
  par.D_ref = q_ref;  //reference drainage rate
  par.s_ref = sc_ref; //reference storage level
  par.b = bc;         //drainage exponent
  par.a = -1;         //value of -1 shuts off bypassing
  par.c = 0.0;        //there's no bypassing of the channel box
  
  //for channel box, pet=0 and p=qGW+qOF+qSS, representing inputs to channel instead of precipitation
  
  
  //now integrate the channel box
  for (i=0; i<n; i++) {
    par.p = qOFa[i]+qSSa[i]+qGWa[i];
    if (i==0) sc_pre = sc_init; else sc_pre = sc[i-1];   //need to do start with sc_init at first time step
    integr(par, sc_pre, dt, tol, sc[i]);  //integrate for new storage value
    
    q[i] = q_ref * pow((sc[i]/sc_ref), bc); //channel discharge at end of time interval
    qa[i] = par.p - (sc[i]-sc_pre)/dt;         //average channel discharge over time interval, calculated from mass balance
  }
  
  return;
  
} //end threebox












///////////////////////////////////////////////
// here is the main routine
///////////////////////////////////////////////


//runs the three-box model, and then perturbs the solution to estimate the runoff response distribution


// [[Rcpp::export]]
List threebox_model(List input)
{
  
  //here we instantiate the inputs into Cpp
  
  int n =              input["n"];       //number of time steps
  double dt =          input["dt"];      //length of time step (for consistency between volumes and rates)
  NumericVector p =    input["p"];       //precipitation rate time series (average rate over time step, not cumulative depth per time step dt)
  NumericVector opet = input["pet"];     //original un-scaled potential evapotranspiration time series (average rate over time step, not cumulative depth per time step dt)
  double su_ref =      input["su_ref"];  //upper box reference volume
  double sl_ref =      input["sl_ref"];  //lower box reference volume
  double sc_ref =      input["sc_ref"];  //channel box reference volume
  double bu =          input["bu"];      //upper box exponent
  double bl =          input["bl"];      //lower box exponent
  double bc =          input["bc"];      //channel box exponent
  double f_OF =        input["f_OF"];    //overland flow fraction of discharge at reference equilibrium
  double f_SS =        input["f_SS"];    //shallow subsurface flow fraction of discharge at reference equilibrium
  double fw =          input["fw"];      //fraction of su_ref over which ET responds to changes in storage
  int npert =          input["npert"];   //length of perturbation time series (in number of time steps)
  int agg =            input["agg"];     //aggregation factor for RRDs
  double tol =         input["tol"];     //precision tolerance for Cash-Karp Runge-Kutta integration
  double pet_mult =    input["pet_mult"]; //multiplier for re-scaling PET
  
  //define vectors and arrays
  
  NumericVector qOF(n);           //overland flow discharge that bypasses upper box (at end of each time step)
  NumericVector qSS(n);           //subsurface flow discharge that bypasses lower box (at end of each time step)
  NumericVector qGW(n);           //lower box discharge (at end of each time step)
  NumericVector q(n);    	        //total discharge (at end of each time step)
  NumericVector qOFa(n);          //overland flow discharge that bypasses upper box, averaged over each time step
  NumericVector qSSa(n);          //subsurface flow discharge that bypasses lower box, averaged over each time step
  NumericVector qGWa(n);          //lower box discharge averaged over each time step
  NumericVector qa(n);    	      //total discharge averaged over each time step
  NumericVector et(n); 	          //actual evapotranspiration, averaged over each time step
  NumericVector R(n); 	          //lower-box recharge, averaged over each time step
  NumericVector su(n);          	//upper box storage (instantaneous at end of each time step)
  NumericVector sl(n);          	//lower box storage (instantaneous at end of each time step)
  NumericVector sc(n);            //channel box storage (instantaneous at end of each time step)
  NumericVector pet(n);           //re-scaled potential evapotranspiration
  NumericVector D(n);             // drainage from upper box (at end of time step)
  NumericVector Da(n);            // drainage from upper box (averaged over time step)
  
  
  NumericVector p_pert(npert);     //perturbed values of the above variables
  NumericVector pet_pert(npert);
  NumericVector qOF_pert(npert);
  NumericVector qSS_pert(npert);
  NumericVector qGW_pert(npert);
  NumericVector q_pert(npert);
  NumericVector qOFa_pert(npert);
  NumericVector qSSa_pert(npert);
  NumericVector qGWa_pert(npert);
  NumericVector qa_pert(npert);
  NumericVector et_pert(npert);
  NumericVector R_pert(npert);
  NumericVector su_pert(npert);
  NumericVector sl_pert(npert);
  NumericVector sc_pert(npert);
  NumericVector D_pert(npert);          
  NumericVector Da_pert(npert);         
  NumericMatrix rrd(n/agg, npert/agg);       //runoff response distribution (note npert/agg is integer division in C...)
  NumericMatrix rrda(n/agg, npert/agg);      //runoff response distribution
  NumericVector agg_q(n/agg);                //instantaneous discharge at aggregated time steps
  NumericVector agg_qa(n/agg);               //averaged discharge at aggregated time steps
  NumericVector agg_p(n/agg);                //averaged precipitation at aggregated time steps
  NumericVector agg_pet(n/agg);              //averaged PET at aggregated time steps
  NumericVector agg_t(n/agg);                //time at end of aggregated time steps (in units of original dt)
  
  
  
  
  int i, j, k, ii, n_rrd, agg_steps, next_time;
  double su_init, sl_init, sc_init, p_bar, pet_bar, D_ref, qGW_ref, q_ref, f_et, s_wp, s_o, aOF, aSS;
  double avgp, avgqa, avgqa_pert;
  
  
  time_t start_time, end_time;  //for elapsed time reporting
  
  time(&start_time);
  
  
  
  ///////  INITIALIZATION   /////////////////////////////////////////////////
  // First we need to initialize the parameters D_ref, qGW_ref, s_wp, s_o, aOF and aSS.
  // These are set so that the system is in equilibrium in the reference state, defined as:
  //    P=P_bar, its long-term average
  //    PET=PET_bar, its long-term average
  //    ET/P follows the Turc-Mezentsev "Budyko curve" with shape factor of n=2
  //    Su=Su_ref
  //    Sl=Sl_ref
  //    qOF/q = f_OF  overland flow is specified fraction of total discharge
  //    qSS/q = f_SS  shallow subsurface flow is specified fraction of total discharge
  //    qGW/q = (1 - f_OF - f_SS) groundwater flow is the rest of total dischrage
  
  
  //first calculate the average P and PET for the whole time series
  p_bar=0.0; for (i=0; i<n; i++) p_bar = p_bar+p[i]; p_bar = p_bar / (double) n;
  pet_bar=0.0; for (i=0; i<n; i++) {
    pet[i] = opet[i]*pet_mult;
    pet_bar = pet_bar+pet[i]; 
  }
  pet_bar = pet_bar / (double) n;
  
  
  //now use the Budyko curve to estimate actual evapotranspiration as a fraction of precipitation
  if (pet_bar<=0.0) f_et = 0.0;
  else f_et = pow( pow(p_bar/pet_bar, 2.0) + 1.0 , -0.5);
  
  
  //now set the wilting point storage parameter s_wp
  if (pet_bar<=0.0) {s_wp=su_ref; pet_bar=0.0;} 
  else  s_wp = su_ref * (1.0 - f_et*p_bar/pet_bar * fw) ;
  if (s_wp<0.0) s_wp=0.0;
  
  //now set the storage at which water stress does not restrict ET
  s_o = s_wp + fw*su_ref;
  
  
  //now set the reference upper-box drainage rate D_ref
  D_ref = (1.0 - f_et)*(1.0 - f_OF)*p_bar;                
  
  
  //now set the set the reference lower-box discharge rate ql_ref
  qGW_ref = (1.0 - f_et)*(1.0 - f_OF - f_SS)*p_bar;
  
  
  //now set the reference channel box discharge rate q_ref
  q_ref = (1.0 - f_et)*p_bar;
  
  
  //now set the exponent aOF (for upper box overland flow partitioning function eta_OF)
  if (f_OF<=0.0) aOF = f_OF;  // negative values are a flag to fix the overland flow fraction rather than make it storage-dependent
  else aOF = log(f_OF*(1.0 - f_et))/log(0.5);
  
  
  //and set the exponent aSS (for the lower box partitioning function eta_SS)
  if (f_SS<=0.0) aSS = f_SS;  // negative or zero values are a flag to fix the shallow subsurface flow fraction rather than make it storage-dependent
  else aSS = log(f_SS/(1.0 - f_OF))/log(0.5);
  
  
  
  
  
  ///////  BASE RUN   /////////////////////////////////////////////////
  //now do the base run of the three-box model
  
  //note we've re-used su_ref and sl_ref for su_init and sl_init, so that the model is initialized in equilibrium
  su_init=su_ref; sl_init=sl_ref; sc_init=sc_ref;  //these are redundant here, but for perturbations we will need to initialize storage at different levels that are not in equilibrium
  
  //now run the three-box model
  threebox(p, pet, n, bu, bl, bc, aOF, aSS, su_ref, sl_ref, sc_ref, su_init, sl_init, sc_init, s_wp, s_o, D_ref, qGW_ref, q_ref, tol, dt, qOF, qSS, qGW, q, qOFa, qSSa, qGWa, qa, et, R, su, sl, sc, D, Da);
  
  
  time(&end_time);
  Rprintf("\n\nBase run done %g seconds\n\n", difftime(end_time,start_time));
  
  
  time(&start_time);
  next_time = 5;
  
  
  
  
  
  ///////  PERTURB PRECIPITATION   /////////////////////////////////////////////////
  //now do perturbations
  
  
  n_rrd = npert/agg;                                 //number of RRD lags (note this is integer division)
  
  
  for (i=0; i<(n-npert); i=i+agg)                    //step by agg
  { avgp = 0.0;                                      // average precipitation over the aggregation interval
    for (ii=0; ii<agg; ii++) avgp = avgp+p[i+ii];
    avgp = avgp/agg;
    
    if (avgp==0.0)             // exclude cases where p==0
      for (k=0; k<n_rrd; k++) {
        rrd(i/agg,k) = NA_REAL;
        rrda(i/agg,k) = NA_REAL;
      }
      
      else {
        
        //copy over the relevant part of the precipitation record
        for (k=0; k<npert; k++) p_pert[k] = p[i+k];
        
        //and the PET record
        for (k=0; k<npert; k++) pet_pert[k] = pet[i+k];
        
        //perturb the copied precipitation record by setting the first agg values equal to zero
        for (ii=0; ii<agg; ii++) p_pert[ii] = 0.0;
        
        //set initial storage values to match base run storage when the perturbation starts
        if (i==0) {su_init=su_ref; sl_init=sl_ref; sc_init=sc_ref;}
        else {su_init=su[i-1]; sl_init=sl[i-1]; sc_init=sc[i-1];}
        
        //run model with perturbed precip record
        threebox(p_pert, pet_pert, npert, bu, bl, bc, aOF, aSS, su_ref, sl_ref, sc_ref, su_init, sl_init, sc_init, s_wp, s_o, D_ref, qGW_ref, q_ref, tol, dt, qOF_pert, qSS_pert, qGW_pert, q_pert, qOFa_pert, qSSa_pert, qGWa_pert, qa_pert, et_pert, R_pert, su_pert, sl_pert, sc_pert, D_pert, Da_pert);
        
        //calculate runoff response distribution from perturbed values
        
        for (k=0; k<n_rrd; k++) {
          
          //runoff response distribution based on instantaneous discharge          
          rrd(i/agg,k) = (q[i+(k+1)*agg-1]-q_pert[(k+1)*agg-1])/avgp/(agg*dt);     
          
          //average qa and qa_pert over the aggregation interval          
          avgqa = 0.0; avgqa_pert = 0.0;              
          for (ii=0; ii<agg; ii++) { avgqa = avgqa + qa[i+k*agg+ii];  avgqa_pert = avgqa_pert + qa_pert[k*agg+ii]; }
          avgqa = avgqa/agg;  avgqa_pert = avgqa_pert / agg;
          
          //runoff response distribution based on time-averaged discharge          
          rrda(i/agg,k) = (avgqa-avgqa_pert)/avgp/(agg*dt);  
          
          if (k==0) rrda(i/agg,k) = 2.0 * rrda(i/agg,k);           //with time-averaged rrd, we need to multiply the value at the first lag by 2 (since it can only react to half of the time step)
          
        }  //next k
      }  //end else
      
      // give a progress report every 5 seconds
      time(&end_time);                                      // check time
      if (difftime(end_time, start_time)>=next_time) {      // if more than 5 seconds have elapsed since last progress report
        Rprintf("perturbations %3.1f percent done at %g seconds   \r", (i+1.0)/(n-npert)*100.0, difftime(end_time,start_time));
        next_time = next_time+5;                            // iterate 5-second counter
      }
      
      
  } //next i
  
  
  //REMEMBER that RRD is not defined for the last npert time steps!
  for (i=(n-npert); i<n; i=i+agg) for (k=0; k<n_rrd; k++) {
    rrd(i/agg,k) = NA_REAL;
    rrda(i/agg,k) = NA_REAL;
  }
  
  
  time(&end_time);
  Rprintf("Perturbations done %g seconds                            \n\n", difftime(end_time,start_time));
  
  
  
  //now aggregate end results
  
  if (agg==1) {
    for (i=0; i<n; i++) {
      agg_t[i] = (i+1)*dt;                  //time at end of aggregated time steps (in units of original dt)
      agg_q[i] = q[i];                  
      agg_qa[i] = qa[i];
      agg_p[i] = p[i];
      agg_pet[i] = opet[i];
    } // next i
    
  } else {
    
    agg_steps = n/agg;
    
    for (i=0; i<agg_steps; i++) {
      agg_t[i] = (i+1)*dt;                  //time at end of aggregated time steps (in units of original dt)
      
      agg_q[i] = q[(i+1)*agg-1];             //just copy over discharge at end of aggregated time step
      
      //for the rest, we need to calculate averages
      agg_qa[i] = 0.0; agg_pet[i] = 0.0; agg_p[i] = 0.0;
      
      for (j=i*agg; j<(i+1)*agg; j++) {
        agg_qa[i] = agg_qa[i]+qa[j];
        agg_p[i] = agg_p[i]+p[j];
        agg_pet[i] = agg_pet[i]+opet[j];
      }
      
      agg_qa[i] = agg_qa[i]/agg; agg_p[i] = agg_p[i]/agg; agg_pet[i] = agg_pet[i]/agg;
      
    } // next i
  } //end if-else
  
  
  
  
  
  
  
  
  //now we compile the output list
  List output;
  output["et"] = et;           //actual evapotranspiration rate (averaged over each time step)
  output["q"] = q;             //total discharge (at end of each time step)
  output["qOF"] = qOF;         //overland flow bypassing upper box (at end of each time step)
  output["qSS"] = qSS;         //shallow subsurface flow bypassing lower box (at end of each time step)
  output["qGW"] = qGW;         //groundwater flow draining from lower box (at end of each time step)
  output["qa"] = qa;           //total discharge (averaged over each time step)
  output["qOFa"] = qOFa;       //overland flow bypassing upper box (averaged over each time step)
  output["qSSa"] = qSSa;       //shallow subsurface flow bypassing lower box (averaged over each time step)
  output["qGWa"] = qGWa;       //groundwater flow draining from lower box (averaged over each time step)
  output["R"] = R;             //lower-box recharge (averaged over each time step)
  output["su"] = su;           //upper box storage (instantaneous value at end of time step)
  output["sl"] = sl;           //lower box storage (instantaneous value at end of time step)
  output["sc"] = sc;           //channel box storage (instantaneous value at end of time step)
  output["D"] = D;             //drainage rate from upper box (instantaneous value at end of time step)
  output["Da"] = Da;           //drainage rate from upper box (averaged over each time step)
  output["rrd"] = rrd;         //runoff response distribution defined on instantaneous discharge at end of each time step
  output["rrda"] = rrda;       //runoff response distribution defined on average discharge in each time step
  output["agg_q"] = agg_q;     //instantaneous discharge at aggregated time steps
  output["agg_qa"] = agg_qa;   //averaged discharge at aggregated time steps
  output["agg_p"] = agg_p;     //averaged precipitation at aggregated time steps
  output["agg_pet"] = agg_pet; //averaged PET at aggregated time steps
  output["agg_p"] = agg_p;     //time at end of aggregated time steps (in units of original dt)
  
  return output;  //and return the output list
  
  
} //end threebox_model




