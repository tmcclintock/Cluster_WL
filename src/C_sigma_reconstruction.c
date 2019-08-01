/** @file C_sigma_reconstruction.c
 *  @brief Reconstructed Sigma(R) profiles from large scales.
 * 
 *  These experimental routines compute the "Y" 
 *  cluster profiles. This is also known as the
 *  Sigma(R) reconstruction profiles.
 *
 *  This file will undergo much cleanup in the near future.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_sigma_reconstruction.h"

#include <stdio.h>

/**
 * \brief Reconstructed Sigma profiles (i.e. Y)
 * from a precomputed DeltaSigma profile.
 * 
 * Note: output Sigma profile has one less element than 
 * the input DeltaSigma, also we assume a regular grid
 * spacing in the ln(R).
 */
int Sigma_REC_from_DeltaSigma(double dlnR, double*DeltaSigma, int N,
			      double*Sigma){
  /*
    The transformation is:

    Y = T*DeltaSigma = (2S - I)*DeltaSigma

    Where I is the identity matrix and S contains the Runge-kutta
    elements for a midpoint integration method (i.e.
    1/2s on the edges and 1s in the middle)
   */
  //Loop over Sigma elements; i.e. rows
  int i, j;
  for(i = 0; i < N-1; i++){
    Sigma[i] = -DeltaSigma[i]; //Not sure if this is positive or negative
    
    //Loop over DeltaSigma elements; i.e. columns
    //for(j = N-2-i; j < N; j++){
    for(j = i; j < N; j++){
      Sigma[i] -= 2*dlnR*DeltaSigma[j];
      //Downweight the first and last contributions (midpoint formula)
      //if ((j == N-2-i) || (j == N-1)){
      if ((j == i) || (j == N-1)){
	Sigma[i] += dlnR*DeltaSigma[j];
      }
      //printf("%e\n",Sigma[i]);
    }
  }
  return 0;
}
