#pragma once

//#define OUTPUT_FT_NOISE
#define EVOLUTION

#define END_Y	10.0
#define OUTPUT_Y	0.1
#define DELTA_Y	0.01/3.0
#define EPS 1.0e-8


//GPU Block size
#define BSZ  32

//You should choose NX the power of 2.
#define NX  256
#define LATTICE_SIZE  8

//The number of the initial 
#define INITIAL_N  200

#define BATCH  10
#define M_PI  3.141592653589793238462643383
#define Nc 3
#define ADJNc 8
#define ALPHA_S	0.3
//#define v_Parameter	0.3

//#define Round_Proton

#define NUMBER512

