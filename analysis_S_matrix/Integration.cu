
#include <cufft.h> 
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <boost/math/special_functions/bessel.hpp>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <cctype>
#include<complex>
#include <functional>

#include "Header_Params.h"
//#include "interpolate_2d_array.h"

//Bessel
__device__ double bessel_ink_0_4(const double x_val)
{
	double bessel_in = 0.0;
	bessel_in = j1(x_val);

	return bessel_in;

}

//Bessel
__device__ double bessel_ink_2_4(const double x_val)
{
	double bessel_in_2 = 0.0;

	if (0 <= x_val && x_val <1e-5) {
		bessel_in_2 = x_val*x_val*x_val / 32.0 - x_val*x_val*x_val*x_val*x_val / 576.0;
	}
	else {

		bessel_in_2 = (2.0 - 2.0*j0(x_val) - x_val*j1(x_val)) / x_val;
	}

	return bessel_in_2;

}


__device__ __host__ double cos_x_y(const double x_1, const double y_1, const double x_2, const double y_2)
{
	if (sqrt(x_1*x_1 + y_1*y_1) < 1.0e-15 || sqrt(x_2*x_2 + y_2*y_2) < 1.0e-15) {
		return 1.0/sqrt(2.0);
	}
	else {
		return 1.0*(x_1*x_2 + y_1*y_2)
			/ sqrt(x_1*x_1 + y_1*y_1)
			/ sqrt(x_2*x_2 + y_2*y_2);
	}
}


__device__ double cos_x_y_test(const double x_1, const double y_1, const double x_2, const double y_2)
{
	auto arccos_cartesian = [](double x, double y) {
		if (abs(x) < 1e-8 && abs(y) <1e-8) {
			return 0.0;
		}
		else if (x / sqrt(x*x + y * y) >1.0) {
			return 0.0;
		}
		else if (x / sqrt(x*x + y * y) < -1.0) {
			return M_PI;
		}
		else if (y>0.0) {
			return acos(x / sqrt(x*x + y * y));
		}
		else {
			return -1.0*acos(x / sqrt(x*x + y * y));
		}
	};


	double phi_1 = arccos_cartesian(x_1, y_1);
	double phi_2 = arccos_cartesian(x_2, y_2);

	if (sqrt(x_1*x_1 + y_1*y_1) < 1.0e-15 || sqrt(x_2*x_2 + y_2*y_2) < 1.0e-15) {
		return cos(phi_1 - phi_2);
	}
	else {
		return cos(phi_1 - phi_2);
	}
}


__device__ double integrate_function_cos_test_with_y(const double x_1, const double y_1, const double x_2, const double y_2)
{
	return cos_x_y(x_1, y_1, x_2, y_2)*cos_x_y(x_1, y_1, x_2, y_2)*exp(-(x_1 - x_2)*(x_1 - x_2) - (y_1 - y_2)*(y_1 - y_2));
}


__device__ double integrate_function_cos_test(const double x_1, const double y_1, const double x_2, const double y_2)
{
	double rerere = cos_x_y(x_1, y_1, x_2, y_2)*cos_x_y(x_1, y_1, x_2, y_2)*exp(-(x_1)*(x_1)-(y_1)*(y_1));
	//double rerere = cos_x_y(x_1, y_1, x_2, y_2)*exp(-(x_1)*(x_1)-(y_1)*(y_1));
	//double rerere = cos_x_y(x_1, y_1, x_2, y_2);
	return rerere;
}

__device__ double integrate_function_exp_test(const double x_1, const double y_1)
{
	return exp(-(x_1)*(x_1)-(y_1)*(y_1));
}

__device__ double test_return_device(const double x_1, const double y_1)
{
	return x_1 + y_1;
}


__global__ void integration_test(double* Function, double* x_1, double* y_1, double h, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	Function[index] = 0;
	//Function[index] += cos_x_y(x_1[index], y_1[index], x_1[index], y_1[index]);
	//Function[index] += cos_x_y(x_1[0], y_1[0], x_1[index], y_1[index]);
	//double yyy1 = y_1[index];

	for (int m = 0; m < N; m++) {
		for (int n = 0; n < N; n++)
		{
			//Function[index] += 1.0;
			Function[index] += integrate_function_cos_test(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]);
			//Function[index] += cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]);
			//Function[index] += cos_x_y(x_1[10], y_1[10], x_1[index], y_1[index])*cos_x_y(x_1[10], y_1[10], x_1[index], y_1[index]);
			//Function[index] += exp(- 2.0*x_1[m * N + n]* x_1[m * N + n] - 2.0*y_1[m * N + n]* y_1[m * N + n]);
			//Function[index] += exp(- 2.0*x_1[index]* x_1[index] - 2.0*y_1[index]* y_1[index]);
			//Function[index] += integrate_function_exp_test(x_1[m * N + n] , y_1[index]);
			//Function[index] += test_return_device(x_1[m * N + n], y_1[index]);
			//Function[index] += x_1[m * N + n]+ y_1[index];
			//Function[index] += cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])
			//					*exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
		}
	}

	Function[index] = Function[index]*h*h;

}

__global__ void test_global(double* Function,double* x_1, double h, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	if (j < N && i < N) {

		//Function[index] = x_1[index];

		for (int m = 0; m < N; ++m) {

			Function[index] += x_1[N*m+1]*x_1[N*m + 1]/h;
		}
	}

}

__global__ void integration_nonE(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, double* x_1, double* y_1, double h, int N_ini, int N_las, int N){

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las-1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N-1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
					cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

					//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
					if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N )*(N )-1)
						&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)){
						//trV_V = make_cuDoubleComplex(3.0, 0.0);
						for (int tr = 0; tr < 3; ++tr) {
								trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + tr]);
						}
					}
					else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1) 
						&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

						for (int tr = 0; tr < 3; ++tr) {
							trV_V = cuCadd(trV_V, cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * tr + tr]));
						}
					}
					else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1 
						|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

						trV_V = make_cuDoubleComplex(3.0, 0.0);
					}
					else {
						//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								trV_V = cuCadd(trV_V, 
								cuCmul(cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							}
						}
					}

					double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
					if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
						//integrated[index] = integrated[index];
					}
					else {
						double real_coeff = 1.0
							*simpson1*simpson2
							*2.0* bessel_ink_0_4(2.0*P_UPPER*relative_distance) * P_UPPER / relative_distance 
							/ Nc
							* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
						cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

						integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
						//integrated[index] = cuCadd(integrated[index], coeff);
					}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}

__global__ void integration_E(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, double* x_1, double* y_1, double h, int N_ini, int N_las, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i; 
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}
				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
					cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

					//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
					if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
						&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
						//trV_V = make_cuDoubleComplex(3.0, 0.0);
						for (int tr = 0; tr < 3; ++tr) {
							trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + tr]);
						}
					}
					else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
						&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

						for (int tr = 0; tr < 3; ++tr) {
							trV_V = cuCadd(trV_V, cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * tr + tr]));
						}
					}
					else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
						|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

						trV_V = make_cuDoubleComplex(3.0, 0.0);
					}
					else {
						//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								trV_V = cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							}
						}
					}

					double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
					if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
						//integrated[index] = integrated[index];
					}
					else {
						double real_coeff = 1.0
							*simpson1*simpson2
							*2.0* bessel_ink_2_4(2.0*P_UPPER*relative_distance) * P_UPPER / relative_distance 
							/ Nc
							* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n])
							*(2.0*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0);
						cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

						integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
						//integrated[index] = cuCadd(integrated[index], coeff);
					}
				//}
			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}

__global__ void integration_E_test(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, double* x_1, double* y_1, double h, int N_ini, int N_las, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0, 0);

		//for (int m = N_ini; m < N_las; m++) {
			int m = 64;
			//for (int n = N/2; n < 3*N/4; n++) {
				int n = 65;
				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
					cuDoubleComplex trV_V = make_cuDoubleComplex(0, 0);

					if ((index + m * N + n) < 0 || (index + m * N + n) > (N)*(N)-1 || (index - m * N - n) < 0 || (index - m * N - n) > (N )*(N )-1) {
						trV_V = make_cuDoubleComplex(0, 0);
					}
					else {
						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								//int tr = 0;
								//int in = 1;
								cuDoubleComplex trV_tes = make_cuDoubleComplex(1, 0);
								cuDoubleComplex trV_tes2 = make_cuDoubleComplex(3 * 3 * (index + m * N + n) + 3 * tr + in, 0);
								trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * (index + m * N + n) + 3 * tr + in]), V_matrix[3 * 3 * (index - m * N - n) + 3 * in + tr]));
								//trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * (index + m * N + n) + 3 * tr + in]), trV_tes));
								//trV_V = cuCadd(trV_V, cuCmul(V_matrix[3 * 3 * (index + m * N + n) + 3 * tr + in], trV_tes));
								//trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (index + m * N + n) + 3 * tr + in]);
								//trV_V =  V_matrix[3 * 3 * (index + m * N + n) + 3 * tr + in];
								//trV_V = V_matrix[3 * 3 * (index + m * N + n) ];

								//trV_V = V_matrix[index + m * N + n];
								//trV_V = cuCadd(trV_V, trV_tes2);
							}
						}
					}

					double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
					if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
					}
					else {
						double real_coeff = 2.0;
						cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0);

						integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
					}
				//}
			//}
		//}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(2.0*h*2.0*h, 0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}

void cos_integration_test(double* integrated_result)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N], *x2 = new double[N*N], *y2 = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			x2[N*j + i] = xmin + i*h;
			y2[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}


	// Allocate arrays on the device
	double *x_d, *y_d, *Integrated_d ;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMalloc((void**)&Integrated_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);


	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	integration_test <<<dimGrid, dimBlock >>> (Integrated_d, x_d, y_d, h, N);
	//test_global <<<dimGrid, dimBlock >>> (Integrated_d, x_d,h, N);

	cudaMemcpy(integrated_result, Integrated_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}

__global__ void test_copy(cuDoubleComplex *comptest_d)
{

	comptest_d[0] = make_cuDoubleComplex(0, 0);
	for (int i = 0; i < 10; ++i) {
		cuDoubleComplex comptemp_d;
		comptemp_d = make_cuDoubleComplex(1.0*i, 2.0*i);
		comptest_d[0] = cuCadd(comptest_d[0], comptemp_d);
	}
}

inline void test_copy_fDtH()
{

	std::complex<double> *comptest = new std::complex<double>[1];

	cuDoubleComplex *comptest_d;
	cudaMalloc((void**)&comptest_d, sizeof(cuDoubleComplex) );
	test_copy <<<1, 1 >>>(comptest_d);
	cudaMemcpy(comptest, comptest_d, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);


	cudaFree(comptest_d);
	delete[](comptest);
}


__global__ void integration_nonE_test(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	cuDoubleComplex testtest = make_cuDoubleComplex(0, 0);

	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);


		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m;
				if (m == 0 || m == N - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				cuDoubleComplex trV_V = make_cuDoubleComplex(1.0, 0);
		//		for (int tr = 0; tr < 3; ++tr) {
		//			for (int in = 0; in < 3; ++in) {
						//int inner = tr + in;
						//cuDoubleComplex testtest2 = make_cuDoubleComplex(3, 1);
		//				cuDoubleComplex tempCV = cuConj(V_matrix[3 * 3 * ( m * N + n) + 3 * tr + in]);
		//				cuDoubleComplex tempCV2 = cuCmul(tempCV, V_matrix[3 * 3 * ( m * N + n) + 3 * in + tr]);
						//cuDoubleComplex tempCV2 = make_cuDoubleComplex(3, 1);
						//tempCV2.x = V_matrix[inner].x*V_matrix[inner].x - V_matrix[inner].y*V_matrix[inner].y;
						//tempCV2.y = V_matrix[inner].x*V_matrix[inner].y + V_matrix[inner].y*V_matrix[inner].x;
						//tempCV2.x = (V_matrix[N+inner].x*V_matrix[N + inner].x - V_matrix[N + inner].y*V_matrix[N + inner].y);
						//tempCV2.y = (V_matrix[N + inner].x*V_matrix[N + inner].y - V_matrix[N + inner].x*V_matrix[N + inner].y);
						//tempCV2.y = 0;
			//			trV_V = cuCadd(trV_V, tempCV2);
			//		}
			//	}

				//double real_coeff = 1.0;
				//cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0);

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				}
				else {
					double real_coeff = 1.0
						*simpson1*simpson2
						*4.0 * jn(2, 2.0*momk*relative_distance)
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				}

				//integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0);
		integrated[index] = cuCmul(integrated[index], coeff2);
		//integrated[index].x *= 2.0*h*2.0*h;
		//integrated[index].y *= 0;

		if (index == 0) {
			printf("%f",integrated[index].x);
		}
	}
}

void integration_nonE_test_nonG(std::complex<double>* integrated,
	double* x_1, double* y_1, double h, int N, double momk) {

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int index = i * N + j;
			integrated[index] = std::complex<double>(0.0, 0.0);

			for (int m = 0; m < N; m++) {
				for (int n = 0; n < N; n++) {
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					int diffinitm = m;
					if (m == 0 || m == N - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (diffinitm % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {

						simpson1 = 4.0 / 3.0;
					}


					if (n == 0 || n == N - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (n % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {

						simpson2 = 4.0 / 3.0;
					}

					std::complex<double> trV_V = std::complex<double>(1.0, 0);

					double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
					if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
					}
					else {
						double real_coeff = 1.0
							*simpson1*simpson2
							*4.0 * jn(2, 2.0*momk*relative_distance)
							* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n])
							*(2.0*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0);
						std::complex<double> coeff = std::complex<double>(real_coeff, 0.0);

						integrated[index] += coeff*trV_V;
					}

					//integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

				}
			}

			std::complex<double> coeff2 = std::complex<double>(h*h, 0);
			integrated[index] = integrated[index]*coeff2;
			//integrated[index].x *= 2.0*h*2.0*h;
			//integrated[index].y *= 0;
		}
	}
}


__global__ void add_integration(cuDoubleComplex* integrated, cuDoubleComplex* integrated1, cuDoubleComplex* integrated2, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	cuDoubleComplex testtest = make_cuDoubleComplex(3, 0);

	if (i < N && j < N) {
		integrated[index] = cuCadd(integrated1[index], integrated2[index]);
	}
}



__global__ void integration_nonE_Wigner(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, 
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0, 0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + tr]);
					}
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * tr + tr]));
					}
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(3.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
							cuCmul(cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				}
				else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance) 
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void integration_nonE_Wigner_wT(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0, 0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex unit_V = make_cuDoubleComplex(-3.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + tr]);
					}
					trV_V = cuCadd(trV_V, unit_V);
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * tr + tr]));
					}
					trV_V = cuCadd(trV_V, unit_V);
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
					trV_V = cuCadd(trV_V, unit_V);
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				}
				else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance)
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void integration_E_Wigner(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, 
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//integrated[index] = make_cuDoubleComplex(1.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}

				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}
				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + tr]);
					}
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					for (int tr = 0; tr < 3; ++tr) {
						trV_V = cuCadd(trV_V, cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * tr + tr]));
					}
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(3.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
							cuCmul(cuConj(V_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), V_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				}
				else {
					double real_coeff = simpson1*simpson2
						*4.0 * jn(2,2.0*momk*relative_distance)
						//*4.0 * jn(2, 2.0*momk*relative_distance)
						//*(relative_distance*relative_distance)
						//*jn(2,2.0)
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n])
						*(2.0*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0)
						//*(2.0*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0)
						//*(2.0*cos_x_y_test(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y_test(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0)
						//*(2.0*cos_x_y_test(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y_test(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0)
						;
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);
					//coeff = make_cuDoubleComplex(1.0, 0.0);
					//trV_V = make_cuDoubleComplex(1, 0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				}
				//}
			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);
		//coeff2 = make_cuDoubleComplex(1.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void Dfferential_U(cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix, cuDoubleComplex* V_matrix,
	 double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	if (i < N && j < N) {

		cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / h, 0.0);
		DxV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
		DyV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
		if (i < N - 1 && j < N - 1) {
			for (int tr = 0; tr < 3; ++tr) {
				for (int in = 0; in < 3; ++in) {

					DxV_matrix[3 * 3 * index + 3 * in + tr] 
						= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
					DyV_matrix[3 * 3 * index + 3 * in + tr] 
						= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
				}
			}

		}
		else if (i == N - 1 && j < N - 1) {

			for (int tr = 0; tr < 3; ++tr) {
				for (int in = 0; in < 3; ++in) {

					DxV_matrix[3 * 3 * index + 3 * in + tr] 
						= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
					if (tr == in) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						DyV_matrix[3 * 3 * index + 3 * in + tr] 
							= cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						DyV_matrix[3 * 3 * index + 3 * in + tr] 
							= cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
					}


				}
			}

		}
		else if (i < N - 1 && j == N - 1) {

			for (int tr = 0; tr < 3; ++tr) {
				for (int in = 0; in < 3; ++in) {

					DyV_matrix[3 * 3 * index + 3 * in + tr] 
						= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
					if (tr == in) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						DxV_matrix[3 * 3 * index + 3 * in + tr] 
							= cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						DxV_matrix[3 * 3 * index + 3 * in + tr] 
							= cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
					}


				}
			}

		}
		else if (i == N - 1 && j == N - 1) {

			for (int tr = 0; tr < 3; ++tr) {
				for (int in = 0; in < 3; ++in) {


					if (tr == in) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						DxV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
						DyV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						DxV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
						DyV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
					}


				}
			}

		}
	}

}

// NX/8 times NX/8 lattice grid
__global__ void Dfferential_U_short(cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix, cuDoubleComplex* V_matrix,
	double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int original_index = j * N + i;

	for (int n = 0; n < 8; ++n) {
		for (int m = 0; m < 8; ++m) {
			int index = (8 * j + n)*N*8 + (8 * i + m);
			if (8 * i + m < NX && 8 * j + n < NX) {

				cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / h, 0.0);
				DxV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
				DyV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);

				if (8 * i + m < NX - 1 && 8 * j + n < NX - 1) {
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {

							DxV_matrix[3 * 3 * index + 3 * in + tr]
								= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + NX) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
							DyV_matrix[3 * 3 * index + 3 * in + tr]
								= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
						}
					}

				}
				else if (8 * i + m == NX - 1 && 8 * j + n < NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {

							DxV_matrix[3 * 3 * index + 3 * in + tr]
								= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + NX) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
							if (tr == in) {
								cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
								DyV_matrix[3 * 3 * index + 3 * in + tr]
									= cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
							}
							else {

								cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
								DyV_matrix[3 * 3 * index + 3 * in + tr]
									= cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
							}


						}
					}

				}
				else if (8 * i + m < NX - 1 && 8 * j + n == NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {

							DyV_matrix[3 * 3 * index + 3 * in + tr]
								= cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * in + tr], V_matrix[3 * 3 * index + 3 * in + tr]));
							if (tr == in) {
								cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
								DxV_matrix[3 * 3 * index + 3 * in + tr]
									= cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
							}
							else {

								cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
								DxV_matrix[3 * 3 * index + 3 * in + tr]
									= cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
							}


						}
					}

				}
				else if (8 * i + m == NX - 1 && 8 * j + n == NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {


							if (tr == in) {
								cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
								DxV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
								DyV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * in + tr]));
							}
							else {

								cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
								DxV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
								DyV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * in + tr]));
							}


						}
					}

				}
			}
		}
	}
}



__global__ void Take_Uzero(cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix, cuDoubleComplex* V_matrix,
	double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	if (i < N  && j < N ) {

		DxV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
		DyV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
	}


}


__global__ void Udagger_Dfferential_U(cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix, cuDoubleComplex* V_matrix,
	 double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / h, 0.0);
	VdDxV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
	VdDyV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
	if (i < N - 1 && j < N - 1) {
		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {

				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

				for (int temp = 0; temp < 3; ++temp) {

					trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
					trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
				}

				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;
			}
		}

	}
	else if (i == N - 1 && j < N - 1) {

		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {

				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);
				for (int temp = 0; temp < 3; ++temp) {

					trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
				}
				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;

				for (int temp = 0; temp < 3; ++temp) {
					if (tr == temp) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
								cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
								cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}

				}
				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;


			}
		}

	}
	else if (i < N - 1 && j == N - 1) {

		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {


				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

				for (int temp = 0; temp < 3; ++temp) {

					trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
				}

				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;

				for (int temp = 0; temp < 3; ++temp) {

					if (tr == temp) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
				}

				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;


			}
		}

	}
	else if (i == N - 1 && j == N - 1) {

		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {

				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

				for (int temp = 0; temp < 3; ++temp) {

					if (tr == temp) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
								cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
								cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}


				}
				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;


			}
		}

	}


}


// NX/8 times NX/8 lattice grid
__global__ void Udagger_Dfferential_U_short(cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix, cuDoubleComplex* V_matrix,
	double h, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int original_index = j * N + i;

	for (int n = 0; n < 8; ++n) {
		for (int m = 0; m < 8; ++m) {
			int index = (8 * j + n)*N * 8 + (8 * i + m);
			if (8 * i + m< NX && 8 * j + n< NX) {

				cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / h, 0.0);
				VdDxV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);
				VdDyV_matrix[index] = make_cuDoubleComplex(0.0, 0.0);

				if (8 * i + m< NX - 1 && 8 * j + n< NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
							cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

							for (int temp = 0; temp < 3; ++temp) {

								trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
									cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
								trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
									cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
							}

							VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
							VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;
						}
					}


				}
				else if (8 * i + m == NX - 1 && 8 * j + n< NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
							cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);
							for (int temp = 0; temp < 3; ++temp) {

								trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
									cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
							}
							VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;

							for (int temp = 0; temp < 3; ++temp) {
								if (tr == temp) {
									cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
									trV_V2
										= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
											cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}
								else {

									cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
									trV_V2
										= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
											cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}

							}
							VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;

						}
					}

				}
				else if (8 * i + m < NX - 1 && 8 * j + n == NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
							cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

							for (int temp = 0; temp < 3; ++temp) {

								trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
									cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
							}

							VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;

							for (int temp = 0; temp < 3; ++temp) {

								if (tr == temp) {
									cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
									trV_V
										= cuCadd(trV_V,
											cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}
								else {

									cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
									trV_V
										= cuCadd(trV_V,
											cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}
							}

							VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
						}
					}

				}
				else if (8 * i + m == NX - 1 && 8 * j + n == NX - 1) {

					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {

							cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
							cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

							for (int temp = 0; temp < 3; ++temp) {

								if (tr == temp) {
									cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
									trV_V
										= cuCadd(trV_V,
											cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
									trV_V2
										= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
											cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}
								else {

									cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
									trV_V
										= cuCadd(trV_V,
											cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
									trV_V2
										= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * in + temp]),
											cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
								}


							}
							VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
							VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;
						}

					}
				}
			}
		}
	}
}


__global__ void nonE_Wigner(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), 
									DxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]),
									DyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				//}
				//else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance)
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h/ALPHA_S/M_PI, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void nonE_Wigner_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N && (i == j || (N - i == j) || (i == N - j))) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]),
									DxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]),
									DyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				double real_coeff = simpson1*simpson2
					*4.0 * j0(2.0*momk*relative_distance)
					* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
				cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}

__global__ void nonE_Wigner_diagonal_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	for (int n = 0; n <= 8; ++n) {
		for (int m = 0; m <= 8; ++m) {
			int index = (8 * j + n)*N + (8 * i + m);
			if (n == 8 && m == 8) { continue; }
			if ((i < N / 8 && j < N / 8 && (8 * i + m) == (8 * j + n))
				|| (i < N/8 && j < N/8 && (8 * i + m) == (N - (8 * j + n)))) {
				integrated[index] = make_cuDoubleComplex(0.0, 0.0);
				//sit the index which is center of the gaussian.
				int recentered_index = index - (N*(N / 2) + N / 2);

				//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
				for (int m = N_ini; m < N_las; m++) {
					for (int n = 0; n < N; n++) {
						double simpson1 = 1.0;
						double simpson2 = 1.0;
						int diffinitm = m - N_ini;
						if (m == N_ini || m == N_las - 1) {
							simpson1 = 1.0 / 3.0;
						}
						else if (diffinitm % 2 == 0) {
							simpson1 = 2.0 / 3.0;
						}
						else {

							simpson1 = 4.0 / 3.0;
						}


						if (n == 0 || n == N - 1) {
							simpson2 = 1.0 / 3.0;
						}
						else if (n % 2 == 0) {
							simpson2 = 2.0 / 3.0;
						}
						else {

							simpson2 = 4.0 / 3.0;
						}

						//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
						//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
						cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

						//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
						if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
							&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
							//trV_V = make_cuDoubleComplex(3.0, 0.0);
							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
							&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
							|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else {
							//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
							for (int tr = 0; tr < 3; ++tr) {
								for (int in = 0; in < 3; ++in) {
									trV_V = cuCadd(trV_V,
										cuCmul(cuConj(DxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]),
											DxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
									trV_V = cuCadd(trV_V,
										cuCmul(cuConj(DyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]),
											DyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
								}
							}
						}

						double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
						//if (relative_distance < 1.0e-10) {
							//integrated[index] += 0;
						//}
						//else {
							double real_coeff = simpson1*simpson2
								*4.0 * j0(2.0*momk*relative_distance)
								* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
							cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

							integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
						//}
						//}

					}
				}

				cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);

				integrated[index] = cuCmul(integrated[index], coeff2);
			}
		}
	}
}


__global__ void nonE_Wigner_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N*8 + i;
	int indexnormal = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index*8 - (NX*(NX / 2) + NX / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = 0; m < NX; m++) {
			for (int n = 0; n < NX; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m ;
				if (m == 0 || m == NX - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == NX - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1)
					&& ((recentered_index + (NX - m) * NX + NX - n) >= 0 && (recentered_index + (NX - m) * NX + NX - n) <= (NX)*(NX)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if (((recentered_index + m * NX + n) >= 0 && (recentered_index + m * NX + n) <= (NX)*(NX)-1)
					&& ((recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1)) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if ((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1
					|| (recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DxV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * in + tr]),
									DxV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * in + tr]));
						}
					}


					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DyV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * in + tr]),
									DyV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * NX + n] * x_1[m * NX + n] + y_1[m * NX + n] * y_1[m * NX + n]);
				if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				}
				else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance)
						* exp(-x_1[m * NX + n] * x_1[m * NX + n] - y_1[m * NX + n] * y_1[m * NX + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[indexnormal] = cuCadd(integrated[indexnormal], cuCmul(coeff, trV_V));
				}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);

		integrated[indexnormal] = cuCmul(integrated[indexnormal], coeff2);
	}
}


__global__ void trU_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N * 8 + i;
	int indexnormal = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index * 8 ;

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						//for (int in = 0; in < 3; ++in) {
						//	trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (recentered_index ) + 3 * in + tr]);
						//}
						trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (recentered_index) + 3 * tr + tr]);
					}

					for (int tr = 0; tr < 3; ++tr) {
						//for (int in = 0; in < 3; ++in) {
						//	trV_V = cuCadd(trV_V, DyV_matrix[3 * 3 * (recentered_index) + 3 * in + tr]);
						//}
						trV_V = cuCadd(trV_V, DyV_matrix[3 * 3 * (recentered_index) + 3 * tr + tr]);
					}
				
					double real_coeff = 1.0;
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[indexnormal] = cuCadd(integrated[indexnormal], cuCmul(coeff, trV_V));

		//cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);
		cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0/6.0, 0.0);

		integrated[indexnormal] = cuCmul(integrated[indexnormal], coeff2);
	}
}


__global__ void nonE_WWWigner_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N*8 + i;
	int indexnormal = j * N  + i;
	if (i < N && j < N) {
		//if (i == j || (N - i) == j || i == (N - j)) {
			integrated[index] = make_cuDoubleComplex(0.0, 0.0);
			//sit the index which is center of the gaussian.
			int recentered_index = index * 8 - (NX*(NX / 2) + NX / 2);

			//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
			for (int m = 0; m < NX; m++) {
				for (int n = 0; n < NX; n++) {
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					int diffinitm = m;
					if (m == 0 || m == NX - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (diffinitm % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {

						simpson1 = 4.0 / 3.0;
					}


					if (n == 0 || n == NX - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (n % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {

						simpson2 = 4.0 / 3.0;
					}

					//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
					//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
					cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

					//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
					if (((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1)
						&& ((recentered_index + (NX - m) * NX + NX - n) >= 0 && (recentered_index + (NX - m) * NX + NX - n) <= (NX)*(NX)-1)) {
						//trV_V = make_cuDoubleComplex(3.0, 0.0);
						trV_V = make_cuDoubleComplex(0.0, 0.0);
					}
					else if (((recentered_index + m * NX + n) >= 0 && (recentered_index + m * NX + n) <= (NX)*(NX)-1)
						&& ((recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1)) {

						trV_V = make_cuDoubleComplex(0.0, 0.0);
					}
					else if ((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1
						|| (recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1) {

						trV_V = make_cuDoubleComplex(0.0, 0.0);
					}
					else {
						//tr(V V) = sum_i sum_j V_{j i} V_{i j} 
						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								trV_V = cuCadd(trV_V,
									cuCmul(DxV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * in + tr],
										DxV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * tr + in]));
							}
						}

						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								trV_V = cuCadd(trV_V,
									cuCmul(DyV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * in + tr],
										DyV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * tr + in]));
							}
						}

					}

					double relative_distance = sqrt(x_1[m * NX + n] * x_1[m * NX + n] + y_1[m * NX + n] * y_1[m * NX + n]);
					//if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
					//}
					//else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance)
						* exp(-x_1[m * NX + n] * x_1[m * NX + n] - y_1[m * NX + n] * y_1[m * NX + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[indexnormal] = cuCadd(integrated[indexnormal], cuCmul(coeff, trV_V));
					//}
					//}

				}
			}

			cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI, 0.0);

			integrated[indexnormal] = cuCmul(integrated[indexnormal], coeff2);
		//}
	}
}


__global__ void nonE_WWWigner(cuDoubleComplex* integrated, cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(VdDxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							trV_V = cuCadd(trV_V,
								cuCmul(VdDyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
					//integrated[index] += 0;
				//}
				//else {
					double real_coeff = simpson1*simpson2
						*4.0 * j0(2.0*momk*relative_distance)
						* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
					cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

					integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}

__global__ void nonE_WWWigner_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N &&(i==j || (N - i ==j) || (i==N-j)) ) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index - (N*(N / 2) + N / 2);

		//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
		for (int m = N_ini; m < N_las; m++) {
			for (int n = 0; n < N; n++) {
				double simpson1 = 1.0;
				double simpson2 = 1.0;
				int diffinitm = m - N_ini;
				if (m == N_ini || m == N_las - 1) {
					simpson1 = 1.0 / 3.0;
				}
				else if (diffinitm % 2 == 0) {
					simpson1 = 2.0 / 3.0;
				}
				else {

					simpson1 = 4.0 / 3.0;
				}


				if (n == 0 || n == N - 1) {
					simpson2 = 1.0 / 3.0;
				}
				else if (n % 2 == 0) {
					simpson2 = 2.0 / 3.0;
				}
				else {

					simpson2 = 4.0 / 3.0;
				}

				//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
				//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

				//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
				if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
					//trV_V = make_cuDoubleComplex(3.0, 0.0);
					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
					&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
					|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

					trV_V = make_cuDoubleComplex(0.0, 0.0);
				}
				else {
					//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
					for (int tr = 0; tr < 3; ++tr) {
						for (int in = 0; in < 3; ++in) {
							trV_V = cuCadd(trV_V,
								cuCmul(VdDxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
							trV_V = cuCadd(trV_V,
								cuCmul(VdDyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				double real_coeff = simpson1*simpson2
					*4.0 * j0(2.0*momk*relative_distance)
					* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
				cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}



__global__ void nonE_WWWigner_diagonal_short(cuDoubleComplex* integrated, cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	for (int n = 0; n <= 8; ++n) {
		for (int m = 0; m <= 8; ++m) {
			int index = (8 * j + n)*N + (8 * i + m);
			if (n == 8 && m == 8) { continue; }
			if ((i < N/8 && j < N/8 && (8 * i + m) == (8 * j + n)) 
				|| (i < N/8 && j < N/8 && (8 * i + m) == (N - (8 * j + n)))) {
				integrated[index] = make_cuDoubleComplex(0.0, 0.0);
				//sit the index which is center of the gaussian.
				int recentered_index = index - (N*(N / 2) + N / 2);

				//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
				for (int m = N_ini; m < N_las; m++) {
					for (int n = 0; n < N; n++) {
						double simpson1 = 1.0;
						double simpson2 = 1.0;
						int diffinitm = m - N_ini;
						if (m == N_ini || m == N_las - 1) {
							simpson1 = 1.0 / 3.0;
						}
						else if (diffinitm % 2 == 0) {
							simpson1 = 2.0 / 3.0;
						}
						else {

							simpson1 = 4.0 / 3.0;
						}


						if (n == 0 || n == N - 1) {
							simpson2 = 1.0 / 3.0;
						}
						else if (n % 2 == 0) {
							simpson2 = 2.0 / 3.0;
						}
						else {

							simpson2 = 4.0 / 3.0;
						}

						//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
						//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
						cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

						//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
						if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
							&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
							//trV_V = make_cuDoubleComplex(3.0, 0.0);
							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
							&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else if ((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1
							|| (recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1) {

							trV_V = make_cuDoubleComplex(0.0, 0.0);
						}
						else {
							//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
							for (int tr = 0; tr < 3; ++tr) {
								for (int in = 0; in < 3; ++in) {
									trV_V = cuCadd(trV_V,
										cuCmul(VdDxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
											VdDxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
									trV_V = cuCadd(trV_V,
										cuCmul(VdDyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
											VdDyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
								}
							}
						}

						double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
						//if (relative_distance < 1.0e-10) {
							//integrated[index] += 0;
						//}
						//else {
							double real_coeff = simpson1*simpson2
								*4.0 * j0(2.0*momk*relative_distance)
								* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
							cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

							integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
						//}
						//}

					}
				}

				cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI, 0.0);

				integrated[index] = cuCmul(integrated[index], coeff2);
			}
		}
	}
}


__global__ void nonE_Smatrix_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N * 8 + i;
	int indexnormal = j * N + i;
	if (i < N && j < N) {
		//if (i == j || (N - i) == j || i == (N - j)) {
			integrated[index] = make_cuDoubleComplex(0.0, 0.0);
			//sit the index which is center of the gaussian.
			int recentered_index = index * 8 - (NX*(NX / 2) + NX / 2);

			//If x=N*j+i, then -x=N*(N-j)+N-i(when the origin is x= N*N/2 + N/2).
			for (int m = 0; m < NX; m++) {
				for (int n = 0; n < NX; n++) {
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					int diffinitm = m;
					if (m == 0 || m == NX - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (diffinitm % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {

						simpson1 = 4.0 / 3.0;
					}


					if (n == 0 || n == NX - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (n % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {

						simpson2 = 4.0 / 3.0;
					}

					//if (abs(x_1[index] + x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] + y_1[m * N + n]) < LATTICE_SIZE / 2
					//	&& abs(x_1[index] - x_1[m* N + n]) < LATTICE_SIZE / 2 && abs(y_1[index] - y_1[m * N + n]) < LATTICE_SIZE / 2) {
					cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

					//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
					if (((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1)
						&& ((recentered_index + (NX - m) * NX + NX - n) >= 0 && (recentered_index + (NX - m) * NX + NX - n) <= (NX)*(NX)-1)) {
						//trV_V = make_cuDoubleComplex(3.0, 0.0);
						for (int tr = 0; tr < 3; ++tr) {
							trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * tr + tr]);
						}
					}
					else if (((recentered_index + m * NX + n) >= 0 && (recentered_index + m * NX + n) <= (NX)*(NX)-1)
						&& ((recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1)) {

						for (int tr = 0; tr < 3; ++tr) {
							trV_V = cuCadd(trV_V, cuConj(DxV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * tr + tr]));
						}
					}
					else if ((recentered_index + m * NX + n) < 0 || (recentered_index + m * NX + n) > (NX)*(NX)-1
						|| (recentered_index + (NX - m) * NX + NX - n) < 0 || (recentered_index + (NX - m) * NX + NX - n) > (NX)*(NX)-1) {

						trV_V = make_cuDoubleComplex(3.0, 0.0);
					}
					else {
						//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
						for (int tr = 0; tr < 3; ++tr) {
							for (int in = 0; in < 3; ++in) {
								trV_V = cuCadd(trV_V,
									cuCmul(cuConj(DxV_matrix[3 * 3 * (recentered_index + m * NX + n) + 3 * in + tr]),
										DxV_matrix[3 * 3 * (recentered_index + (NX - m) * NX + NX - n) + 3 * in + tr]));
							}
						}
					}

					double relative_distance = sqrt(x_1[m * NX + n] * x_1[m * NX + n] + y_1[m * NX + n] * y_1[m * NX + n]);
					if (relative_distance < 1.0e-10) {
						//integrated[index] += 0;
					}
					else {
						double real_coeff = simpson1*simpson2
							*4.0 * j0(2.0*momk*relative_distance)
							* exp(-x_1[m * NX + n] * x_1[m * NX + n] - y_1[m * NX + n] * y_1[m * NX + n]);
						cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

						integrated[indexnormal] = cuCadd(integrated[indexnormal], cuCmul(coeff, trV_V));
					}
					//}

				}
			}

			cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);

			integrated[indexnormal] = cuCmul(integrated[indexnormal], coeff2);
		//}
	}
}


__global__ void Umatrix_x_short(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N * 8 + i;
	int indexnormal = j * N + i;
	if (i < N && j < N) {

		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.
		int recentered_index = index * 8 - (NX*(NX / 2) + NX / 2);

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);

		integrated[indexnormal] = cuCmul(integrated[indexnormal], DxV_matrix[recentered_index]);
	}
}


__global__ void test_bessel(double* number, int N, double momk) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	number[index] = jn(2, 3.0*momk);

}

void integration_nonElliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_result)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}



	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex)*3*3*N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 *N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d,*Integrated_half1_d, *Integrated_half2_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_half1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_half2_d, sizeof(cuDoubleComplex)*N*N);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	//std::vector<std::complex<double>> integ(N*N, 0);

	//integration_nonE_test <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, N);
	//integration_nonE_test_half2 <<<dimGrid, dimBlock >>> (Integrated_half2_d, V_matrix_d, x_d, y_d, h, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_half1_d, Integrated_half2_d, N);

	cuDoubleComplex *Integrated_temp1_d, *Integrated_temp2_d;
	cudaMalloc((void**)&Integrated_temp1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_temp2_d, sizeof(cuDoubleComplex)*N*N);

	integration_nonE <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, 0, N / 4, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, N / 4, N / 2, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, N / 2, 3 * N / 4, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, 3 * N / 4, N, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);

	cudaMemcpy(integrated_result, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated_half1_d);
	cudaFree(Integrated_half2_d);
	cudaFree(V_matrix_d);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}

void integration_Elliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_result) {
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}


	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);


	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 *N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *Integrated_temp1_d, *Integrated_temp2_d;
	cudaMalloc((void**)&Integrated_temp1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_temp2_d, sizeof(cuDoubleComplex)*N*N);


	integration_E <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, 0, N/4, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, N / 4, N/2, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, N / 2, 3*N / 4, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, 3 * N / 4, N, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);


	cudaMemcpy(integrated_result, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated_temp1_d);
	cudaFree(Integrated_temp2_d);
	cudaFree(V_matrix_d);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}


void integration_nonElliptic_Wigner(std::complex<double>* V_matrix, std::complex<double>* integrated_result, double mom_k)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}



	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated_half1_d, *Integrated_half2_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_half1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_half2_d, sizeof(cuDoubleComplex)*N*N);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	//std::vector<std::complex<double>> integ(N*N, 0);

	//integration_nonE_test <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, N);
	//integration_nonE_test_half2 <<<dimGrid, dimBlock >>> (Integrated_half2_d, V_matrix_d, x_d, y_d, h, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_half1_d, Integrated_half2_d, N);

	cuDoubleComplex *Integrated_temp1_d, *Integrated_temp2_d;
	cudaMalloc((void**)&Integrated_temp1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_temp2_d, sizeof(cuDoubleComplex)*N*N);

	//integration_nonE_Wigner <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N, mom_k);
	integration_nonE_Wigner_wT <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N, mom_k);
	//integration_nonE_test <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, N, mom_k);
	//integration_nonE_test_nonG(integrated_result, x, y, h, N, mom_k);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, 0, N / 4, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, N / 4, N / 2, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, N / 2, 3 * N / 4, N);
	//integration_nonE <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, 3 * N / 4, N, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);

	cudaMemcpy(integrated_result, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated_half1_d);
	cudaFree(Integrated_half2_d);
	cudaFree(Integrated_temp1_d);
	cudaFree(Integrated_temp2_d);
	cudaFree(V_matrix_d);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}

void integration_Elliptic_Wigner(std::complex<double>* V_matrix, std::complex<double>* integrated_result, double mom_k)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}


	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);


	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *Integrated_temp1_d, *Integrated_temp2_d;
	cudaMalloc((void**)&Integrated_temp1_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated_temp2_d, sizeof(cuDoubleComplex)*N*N);


	//integration_E_Wigner <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N, mom_k);
	integration_nonE_test <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, N, mom_k);
	//integration_nonE_test_nonG(integrated_result, x, y, h, N, P_UPPER);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, 0, N/4, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, N / 4, N/2, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp1_d, V_matrix_d, x_d, y_d, h, N / 2, 3*N / 4, N);
	//integration_E <<<dimGrid, dimBlock >>> (Integrated_temp2_d, V_matrix_d, x_d, y_d, h, 3 * N / 4, N, N);
	//add_integration <<<dimGrid, dimBlock >>> (Integrated_d, Integrated_temp1_d, Integrated_temp2_d, N);

	//double testtttjn = jn(2, 3.0);

	//double *testjn = new double[N*N], *testjn_d;
	//cudaMalloc((void**)&testjn_d, sizeof(double)*N*N);
	//test_bessel <<<dimGrid, dimBlock >>> (testjn_d, N, mom_k);
	//cudaMemcpy(testjn, testjn_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	cudaMemcpy(integrated_result, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	//cudaFree(testjn_d);
	//delete[](testjn);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated_temp1_d);
	cudaFree(Integrated_temp2_d);
	cudaFree(V_matrix_d);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}


void nonElliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}



	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	//dim3 dimBlock(BSZ, BSZ);
	dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix;
	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 *N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 *N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);



	nonE_Wigner_diagonal <<<dimGrid, dimBlock >>> (Integrated_d, DxV_matrix, DyV_matrix, x_d, y_d, h, 0, N, N, mom_k);

	cudaMemcpy(integrated_resultDP, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	nonE_WWWigner_diagonal <<<dimGrid, dimBlock >>> (Integrated2_d, VdDxV_matrix, VdDyV_matrix, x_d, y_d, h, 0, N, N, mom_k);

	cudaMemcpy(integrated_resultWW, Integrated2_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(VdDxV_matrix);
	cudaFree(VdDyV_matrix);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}

void nonElliptic_short(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}



	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N/8*N/8);

	dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N  - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix;
	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);



	//nonE_Wigner_short <<<dimGridS, dimBlockS >>> (Integrated_d, DxV_matrix, DyV_matrix, x_d, y_d, h, N/8, mom_k);
	nonE_WWWigner_short <<<dimGridS, dimBlockS >>> (Integrated_d, DxV_matrix, DyV_matrix, x_d, y_d, h, N/8, mom_k);
	//trU_short <<<dimGridS, dimBlockS >>> (Integrated_d, DxV_matrix, DyV_matrix, x_d, y_d, h, N/8, mom_k);

	cudaMemcpy(integrated_resultDP, Integrated_d, sizeof(std::complex<double>)*N/8*N/8, cudaMemcpyDeviceToHost);

	nonE_WWWigner_short <<<dimGridS, dimBlockS >>> (Integrated_d, VdDxV_matrix, VdDyV_matrix, x_d, y_d, h, N/8, mom_k);
	//trU_short <<<dimGridS, dimBlockS >>> (Integrated_d, VdDxV_matrix, VdDyV_matrix, x_d, y_d, h, N/8, mom_k);

	cudaMemcpy(integrated_resultWW, Integrated_d, sizeof(std::complex<double>)*N/8*N/8, cudaMemcpyDeviceToHost);


	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(V_matrix_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(VdDxV_matrix);
	cudaFree(VdDyV_matrix);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}


void Smatrix_short(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}



	// Allocate arrays on the device
	double *x_d, *y_d;
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N / 8 * N / 8);

	dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix;
	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N / 8);
	//Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N / 8);
	//Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);



	nonE_Smatrix_short <<<dimGridS, dimBlockS >>> (Integrated_d, V_matrix_d, V_matrix_d, x_d, y_d, h, N / 8, mom_k);

	cudaMemcpy(integrated_resultDP, Integrated_d, sizeof(std::complex<double>)*N / 8 * N / 8, cudaMemcpyDeviceToHost);

	nonE_WWWigner_short <<<dimGridS, dimBlockS >>> (Integrated_d, V_matrix_d, V_matrix_d, x_d, y_d, h, N / 8, mom_k);

	cudaMemcpy(integrated_resultWW, Integrated_d, sizeof(std::complex<double>)*N / 8 * N / 8, cudaMemcpyDeviceToHost);


	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(Integrated_d);
	cudaFree(V_matrix_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(VdDxV_matrix);
	cudaFree(VdDyV_matrix);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
}

struct random_point {
	__host__ __device__
		double operator()() {
		// create a minstd_rand object to act as our source of randomness
		thrust::default_random_engine rng;
		// create a normal_distribution to produce floats from the Normal distribution
		// with mean 0.0 and standard deviation 1.0
		thrust::random::normal_distribution<double> dist(0.0, 1.0);

		return dist(rng);
	}
};

//return the square
struct square {
	__host__ __device__ 
		double operator()(double x) {
		return x*x;
	}
};

struct arccos_function {
	__host__ __device__
		double operator()(double x, double y) {
		if (abs(x) < 1e-8 && abs(y) <1e-8) { 
			return 0; 
		}else if (y / sqrt(x*x + y*y) >1.0) {
			return 0;
		}
		else if (y / sqrt(x*x + y*y) < -1.0) {
			return M_PI;
		}
		else {
			return acos(y / sqrt(x*x + y*y));
		}
	}
};


