// This software contains source code provided by NVIDIA Corporation.
// http://docs.nvidia.com/cuda/eula/index.html#nvidia-cuda-samples-end-user-license-agreement

#include <cufft.h> 
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuComplex.h"
#include <boost/math/special_functions/bessel.hpp>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <time.h>
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
#include <complex>
#include <functional>
#include <chrono>

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
	if (i < N - 1 && j < N - 1) {
		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {

				cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
				cuDoubleComplex trV_V2 = make_cuDoubleComplex(0.0, 0.0);

				for (int temp = 0; temp < 3; ++temp) {

					trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp  + in]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
					trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
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

					trV_V = cuCadd(trV_V, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + N) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
				}
				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;

				for (int temp = 0; temp < 3; ++temp) {
					if (tr == temp) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
								cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
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

					trV_V2 = cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
						cuCmul(coeff2, cuCsub(V_matrix[3 * 3 * (index + 1) + 3 * temp + tr], V_matrix[3 * 3 * index + 3 * temp + tr]))));
				}

				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;

				for (int temp = 0; temp < 3; ++temp) {

					if (tr == temp) {
						cuDoubleComplex Unit = make_cuDoubleComplex(1.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
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
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]), cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
								cuCmul(coeff2, cuCsub(Unit, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}
					else {

						cuDoubleComplex Unit0 = make_cuDoubleComplex(0.0, 0.0);
						trV_V
							= cuCadd(trV_V,
								cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]), cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
						trV_V2
							= cuCadd(trV_V2, cuCmul(cuConj(V_matrix[3 * 3 * index + 3 * temp + in]),
								cuCmul(coeff2, cuCsub(Unit0, V_matrix[3 * 3 * index + 3 * temp + tr]))));
					}


				}
				VdDxV_matrix[3 * 3 * index + 3 * in + tr] = trV_V;
				VdDyV_matrix[3 * 3 * index + 3 * in + tr] = trV_V2;


			}
		}

	}


}


__global__ void Udagger_Dfferential_U_make_unitarity(cuDoubleComplex* uVdDxV_matrix, cuDoubleComplex* uVdDyV_matrix,
	cuDoubleComplex* VdDxV_matrix, cuDoubleComplex* VdDyV_matrix, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;

	cuDoubleComplex coeff = make_cuDoubleComplex(1.0 / 2.0, 0.0);
	if (i < N && j < N) {
		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {


				uVdDxV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff,
					cuCsub(VdDxV_matrix[3 * 3 * index + 3 * in + tr],cuConj(VdDxV_matrix[3 * 3 * index + 3 * tr + in])));
				uVdDyV_matrix[3 * 3 * index + 3 * in + tr] = cuCmul(coeff, 
					cuCsub(VdDyV_matrix[3 * 3 * index + 3 * in + tr], cuConj(VdDyV_matrix[3 * 3 * index + 3 * tr + in])));
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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h/ALPHA_S/M_PI / M_PI / 2.0, 0.0);

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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void Wigner_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N) {

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
					//V^dagger_ba V_ab 
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DxV_matrix[recentered_index + m * N + n]),
									DxV_matrix[recentered_index + (N - m) * N + N - n]));
							trV_V = cuCadd(trV_V,
								cuCmul(cuConj(DyV_matrix[recentered_index + m * N + n]),
									DyV_matrix[recentered_index + (N - m) * N + N - n]));
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				double real_coeff = simpson1*simpson2
					*4.0
					* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
				cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void TMD_direct(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix,
	double* x_1, double* y_1, double h, int N_ini, int N_las, int N, double *k) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N && (i == j || (N - i == j) || (i == N - j))) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.

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

							trV_V = cuCadd(trV_V,
								DxV_matrix[ m * N + n]);

				//double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				//double real_coeff = simpson1*simpson2
				//	* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
				cuDoubleComplex coeff = make_cuDoubleComplex(simpson1*simpson2*cos(k[i]*x_1[m * N + n]+k[j]*y_1[m * N + n]),
					simpson1*simpson2*sin(k[i] * x_1[m * N + n] + k[j] * y_1[m * N + n]));

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void Integration_ksp_direct(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix,
	double h, int N_ini, int N_las, int N, double *k) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N && (i == j || (N - i == j) || (i == N - j))) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.

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

				trV_V = cuCadd(trV_V,
					DxV_matrix[m * N + n]);

				//double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				//double real_coeff = simpson1*simpson2
				//	* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n]);
				cuDoubleComplex coeff = make_cuDoubleComplex(simpson1*simpson2,
					simpson1*simpson2);

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void nonE_Smatrix_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix,
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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI / M_PI/2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void E_Smatrix_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix,
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
						}
					}
				}

				double relative_distance = sqrt(x_1[m * N + n] * x_1[m * N + n] + y_1[m * N + n] * y_1[m * N + n]);
				//if (relative_distance < 1.0e-10) {
				//integrated[index] += 0;
				//}
				//else {
				double real_coeff = simpson1*simpson2
					*4.0 * jn(2,2.0*momk*relative_distance)
					* exp(-x_1[m * N + n] * x_1[m * N + n] - y_1[m * N + n] * y_1[m * N + n])
					*(2.0*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index])*cos_x_y(x_1[m * N + n], y_1[m * N + n], x_1[index], y_1[index]) - 1.0);
				if (abs(momk) < 1.0e-8) { real_coeff = 0; }
				cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

				integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));
				//}
				//}

			}
		}

		cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
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

__global__ void trU_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	int indexnormal = j * N + i;
	if (i < N && j < N && (i == j || (N - i == j) || (i == N - j))) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);

		cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

		//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
		for (int tr = 0; tr < 3; ++tr) {
			//for (int in = 0; in < 3; ++in) {
			//	trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (recentered_index ) + 3 * in + tr]);
			//}
			trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (indexnormal)+3 * tr + tr]);
		}

		for (int tr = 0; tr < 3; ++tr) {
			//for (int in = 0; in < 3; ++in) {
			//	trV_V = cuCadd(trV_V, DyV_matrix[3 * 3 * (recentered_index) + 3 * in + tr]);
			//}
			trV_V = cuCadd(trV_V, DyV_matrix[3 * 3 * (indexnormal)+3 * tr + tr]);
		}

		double real_coeff = 1.0;
		cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

		integrated[indexnormal] = cuCadd(integrated[indexnormal], cuCmul(coeff, trV_V));

		//cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);
		cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / 6.0, 0.0);

		integrated[indexnormal] = cuCmul(integrated[indexnormal], coeff2);
	}
}


__global__ void trU_trudagger(cuDoubleComplex* integrated, cuDoubleComplex* DxV_matrix, cuDoubleComplex* DyV_matrix, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N ) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);

		cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

		//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {
				trV_V = cuCadd(trV_V, DxV_matrix[3 * 3 * (index ) + 3 * in + tr]);
				trV_V = cuCadd(trV_V, cuConj(DxV_matrix[3 * 3 * (index)+3 * tr + in]));
			}
		}

		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {
				trV_V = cuCadd(trV_V, DyV_matrix[3 * 3 * (index) + 3 * in + tr]);
				trV_V = cuCadd(trV_V, cuConj(DyV_matrix[3 * 3 * (index)+3 * tr + in]));
			}
		}

		double real_coeff = 1.0;
		cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

		integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

		//cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);
		cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / 18.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void trUUdagger(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N) {
		integrated[index] = make_cuDoubleComplex(0.0, 0.0);

		cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);

		//tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} 
		for (int tr = 0; tr < 3; ++tr) {
			for (int in = 0; in < 3; ++in) {
				trV_V = cuCadd(trV_V, cuCmul(V_matrix[3 * 3 * (index)+3 * in + tr], cuConj(V_matrix[3 * 3 * (index)+3 * in + tr])));
			}
		}


		double real_coeff = 1.0;
		cuDoubleComplex coeff = make_cuDoubleComplex(real_coeff, 0.0);

		integrated[index] = cuCadd(integrated[index], cuCmul(coeff, trV_V));

		//cuDoubleComplex coeff2 = make_cuDoubleComplex(h*h / ALPHA_S / M_PI, 0.0);
		cuDoubleComplex coeff2 = make_cuDoubleComplex(1.0 / 3.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

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
									VdDxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + in]));
							trV_V = cuCadd(trV_V,
								cuCmul(VdDyV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDyV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * tr + in]));
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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
	}
}


__global__ void nonE_WWSmatrix_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* VdDxV_matrix,
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
								cuCmul(VdDxV_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr],
									VdDxV_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
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

		cuDoubleComplex coeff2 = make_cuDoubleComplex(-h*h / ALPHA_S / M_PI / M_PI / 2.0, 0.0);

		integrated[index] = cuCmul(integrated[index], coeff2);
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

__global__ void Umatrix_diagonal(cuDoubleComplex* integrated, cuDoubleComplex* V_matrix,
	double* x_1, double* y_1, double h, int N, double momk) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	int indexnormal = j * N + i;
	if (i < N && j < N && (i == j || (N - i == j) || (i == N - j))) {

		integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		//sit the index which is center of the gaussian.


		integrated[indexnormal] = cuCadd(integrated[indexnormal], V_matrix[indexnormal]);
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
		*f = new double[N*N], *u_a = new double[N*N];
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
	cuDoubleComplex *uVdDxV_matrix, *uVdDyV_matrix;
	cudaMalloc((void**)&uVdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&uVdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);
	Udagger_Dfferential_U_make_unitarity <<<dimGrid, dimBlock >>> (uVdDxV_matrix, uVdDyV_matrix, VdDxV_matrix, VdDyV_matrix, N);



	nonE_Wigner_diagonal <<<dimGrid, dimBlock >>> (Integrated_d, DxV_matrix, DyV_matrix, x_d, y_d, h, 0, N, N, mom_k);
	//trU_trudagger <<<dimGrid, dimBlock >>> (Integrated_d, DxV_matrix, DyV_matrix, N);
	//trUUdagger <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, N);

	cudaMemcpy(integrated_resultDP, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	//nonE_WWWigner_diagonal <<<dimGrid, dimBlock >>> (Integrated2_d, uVdDxV_matrix, uVdDyV_matrix, x_d, y_d, h, 0, N, N, mom_k);
	//u^dDu=-(u^dDu)^d
	nonE_Wigner_diagonal <<<dimGrid, dimBlock >>> (Integrated2_d, uVdDxV_matrix, uVdDyV_matrix, x_d, y_d, h, 0, N, N, mom_k);
	//trU_trudagger <<<dimGrid, dimBlock >>> (Integrated2_d, uVdDxV_matrix, uVdDyV_matrix, N);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

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
	cudaFree(uVdDxV_matrix);
	cudaFree(uVdDyV_matrix);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
}

void Smatrix_value(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k)
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
	//Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	//Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);



	//trU_diagonal <<<dimGrid, dimBlock >>> (Integrated_d, DxV_matrix, DxV_matrix, x_d, y_d, h, N, mom_k);
	nonE_Smatrix_diagonal <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N, mom_k);

	cudaMemcpy(integrated_resultDP, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	//trU_diagonal <<<dimGrid, dimBlock >>> (Integrated2_d, VdDxV_matrix, VdDxV_matrix, x_d, y_d, h, N, mom_k);
	E_Smatrix_diagonal <<<dimGrid, dimBlock >>> (Integrated_d, V_matrix_d, x_d, y_d, h, 0, N, N, mom_k);

	cudaMemcpy(integrated_resultWW, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


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

__global__ void add_k2_sqare_complex_vector(cuDoubleComplex *ft, cuDoubleComplex *ftx, cuDoubleComplex *fty, 
	cuDoubleComplex *Integrated, cuDoubleComplex *Integrated2, double *k_d, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		cuDoubleComplex k2_comp = make_cuDoubleComplex(k_d[j] * k_d[j] + k_d[i] * k_d[i], 0.0);
		//cuDoubleComplex k2_comp = make_cuDoubleComplex(1.0, 0.0);

		Integrated[index]= cuCadd(Integrated[index], cuCmul(k2_comp, cuCmul(cuConj(ft[index]), ft[index])));

		Integrated2[index] = cuCadd(Integrated2[index], cuCmul(cuConj(ftx[index]), ftx[index]));
		Integrated2[index] = cuCadd(Integrated2[index], cuCmul(cuConj(fty[index]), fty[index]));
	}
}


__global__ void add_complex_vector_index( cuDoubleComplex *ftx, cuDoubleComplex *fty,
	cuDoubleComplex *Integrated, cuDoubleComplex *Integrated2, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		Integrated[index] = cuCadd(Integrated[index],  ftx[index]);
		Integrated2[index] = cuCadd(Integrated2[index], fty[index]);
	}
}



__global__ void add_abs_complex_vector(cuDoubleComplex *ftx, cuDoubleComplex *fty,
	cuDoubleComplex *Integrated2,  int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		Integrated2[index] = cuCadd(Integrated2[index], cuCmul(cuConj(ftx[index]), ftx[index]));
		Integrated2[index] = cuCadd(Integrated2[index], cuCmul(cuConj(fty[index]), fty[index]));
	}
}


__global__ void initialize_vectors(cuDoubleComplex *Integrated, cuDoubleComplex *Integrated2, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		Integrated[index] = make_cuDoubleComplex(0.0, 0.0);

		Integrated2[index] = make_cuDoubleComplex(0.0, 0.0);
	}
}

__global__ void index_matrix(cuDoubleComplex *V_index_d, cuDoubleComplex *V_matrix_d, int a, int b, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		V_index_d[index] = V_matrix_d[3 * 3 * index + 3 * a + b];

	}
}

__global__ void index_matrix_subunit(cuDoubleComplex *V_index_d, cuDoubleComplex *V_matrix_d,int a,int b, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		//cuDoubleComplex unit = make_cuDoubleComplex(0.0, 0.0);
		if (a == b) {
			cuDoubleComplex unit = make_cuDoubleComplex(-1.0, 0.0);
			V_index_d[index] = cuCadd(unit, V_matrix_d[3 * 3 * index + 3 * a + b]);
		}
		else {
			cuDoubleComplex unit = make_cuDoubleComplex(0.0, 0.0);
			V_index_d[index] = cuCadd(unit, V_matrix_d[3 * 3 * index + 3 * a + b]);
		}

	}
}

__global__ void index_matrix_Gaussian(cuDoubleComplex *V_index_d, cuDoubleComplex *V_matrix_d, double *x, double *y, int a, int b, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N)
	{
		cuDoubleComplex exponential = make_cuDoubleComplex(exp(-x[index] * x[index] - y[index] * y[index]), 0.0);
		//cuDoubleComplex exponential = make_cuDoubleComplex(-x[i] * x[i] - x[j] * x[j], 0.0);
		//cuDoubleComplex exponential = make_cuDoubleComplex(x[index], 0.0);
		V_index_d[index] = cuCmul(exponential, V_matrix_d[3 * 3 * index + 3 * a + b]);

	}
}


__global__ void index_matrix_fourier(cuDoubleComplex *V_index_d, cuDoubleComplex *V_matrix_d, double *x, double k, int a, int b, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i < N && j < N)
	{
		//cuDoubleComplex exponential = make_cuDoubleComplex(-x[i] * x[i] - x[j] * x[j], 0.0);
		//factor 2 comes from change of variable r to x=2r.
		cuDoubleComplex exponential = make_cuDoubleComplex(cos(2.0*x[index]*k), -sin(2.0*x[index] * k));
		V_index_d[index] = cuCmul(exponential, V_matrix_d[3 * 3 * index + 3 * a + b]);

	}
}


void TMD_value(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}

	cuDoubleComplex test= make_cuDoubleComplex(0.0, 0.0);

	double   *k = new double[N];
	for (int i = 0; i <= N / 2; i++)
	{
		k[i] = i * 2.0 * M_PI / LATTICE_SIZE;
	}
	for (int i = N / 2 + 1; i < N; i++)
	{
		k[i] = (i - N) * 2.0 * M_PI / LATTICE_SIZE;
	}

	// Allocate arrays on the device
	double *k_d ,*x_d, *y_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d, *V_index_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&V_index_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	//dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix, *DxV_index, *DyV_index;
	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&DyV_index, sizeof(cuDoubleComplex)  * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	//cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	//cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	//cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	//Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);

	cuDoubleComplex *ft_d,*ftx_d,*fty_d;
	cudaMalloc((void**)&ft_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&ftx_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&fty_d, sizeof(cuDoubleComplex)*N*N);

	
	initialize_vectors <<<dimGrid, dimBlock >>> (Integrated_d, Integrated2_d, N);

	for (int a = 0; a < Nc; a++) {
		for (int b = 0; b < Nc; b++) {

			index_matrix_subunit <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix,x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix,x_d, y_d, a, b, N);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
				exit(-1);
			}


			cufftHandle plan;
			cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

			cufftExecZ2Z(plan, V_index_d, ft_d, CUFFT_FORWARD);
			cufftExecZ2Z(plan, DxV_index, ftx_d, CUFFT_FORWARD);
			cufftExecZ2Z(plan, DyV_index, fty_d, CUFFT_FORWARD);
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err2));
				exit(-1);
			}


			add_k2_sqare_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, fty_d, Integrated_d, Integrated2_d, k_d, N);
			//for (int i = 0; i < N; i++) {
			//	for (int j = 0; j < N; j++) {
			//		int index = N*j + i;
			//		cuDoubleComplex k2_comp = make_cuDoubleComplex(k_d[j]*k_d[j] + k_d[i]*k_d[i], 0.0);
			//		Integrated_d[index] = cuCadd(Integrated_d[index], cuCmul(k2_comp, cuCmul(cuConj(ft_d[index]), ft_d[index])));
			//		Integrated2_d[index] = cuCadd(Integrated2_d[index], cuCmul(cuConj(ftx_d[index]), ftx_d[index]));
			//		Integrated2_d[index] = cuCadd(Integrated2_d[index], cuCmul(cuConj(fty_d[index]), fty_d[index]));
			//	}
			//}


		}
	}


	cudaMemcpy(integrated_resultDPk, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaMemcpy(integrated_resultDP, Integrated2_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(k_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(ft_d);
	cudaFree(ftx_d);
	cudaFree(fty_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(V_index_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(DxV_index);
	cudaFree(DyV_index);
	//cudaFree(VdDxV_matrix);
	//cudaFree(VdDyV_matrix);
	delete[](k);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
}


void TMD_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 1.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}

	cuDoubleComplex test= make_cuDoubleComplex(0.0, 0.0);

	double   *k = new double[N];
	for (int i = 0; i < NX; i++)
	{
		k[i] = (i-N/2) *2.0*M_PI /1.0 / LATTICE_SIZE;
	}

	// Allocate arrays on the device
	double *k_d ,*x_d, *y_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d, *V_index_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&V_index_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	//dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix, *DxV_index, *DyV_index;

	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&DyV_index, sizeof(cuDoubleComplex)  * N*N);
	cuDoubleComplex *uVdDxV_matrix, *uVdDyV_matrix;
	cudaMalloc((void**)&uVdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&uVdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	//cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix;
	//cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	//cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	//Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Udagger_Dfferential_U_make_unitarity <<<dimGrid, dimBlock >>> (uVdDxV_matrix, uVdDyV_matrix, DxV_matrix, DyV_matrix, N);

	cuDoubleComplex *ft_d,*ftx_d,*fty_d;
	cudaMalloc((void**)&ft_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&ftx_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&fty_d, sizeof(cuDoubleComplex)*N*N);

	
	initialize_vectors <<<dimGrid, dimBlock >>> (Integrated_d, Integrated2_d, N);

	for (int a = 0; a < Nc; a++) {
		for (int b = 0; b < Nc; b++) {
			
			initialize_vectors <<<dimGrid, dimBlock >>> (V_index_d, DxV_index, N);
			initialize_vectors <<<dimGrid, dimBlock >>> (V_index_d, DyV_matrix, N);

			index_matrix_subunit <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);
			//index_matrix <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);
			//index_matrix <<<dimGrid, dimBlock >>> (DxV_index, uVdDxV_matrix, a, b, N);
			//index_matrix <<<dimGrid, dimBlock >>> (DyV_index, uVdDyV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix,x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix,x_d, y_d, a, b, N);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
				exit(-1);
			}


			TMD_direct <<<dimGrid, dimBlock >>> (ft_d, V_index_d, x_d, y_d, h, 0, N, N, k_d);
			TMD_direct <<<dimGrid, dimBlock >>> (ftx_d, DxV_index, x_d, y_d, h, 0, N, N, k_d);
			TMD_direct <<<dimGrid, dimBlock >>> (fty_d, DyV_index, x_d, y_d, h, 0, N, N, k_d);
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed2: %s\n", cudaGetErrorString(err2));
				exit(-1);
			}


			add_k2_sqare_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, fty_d, Integrated_d, Integrated2_d, k_d, N);


		}
	}


	cudaMemcpy(integrated_resultDPk, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaMemcpy(integrated_resultDP, Integrated2_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(k_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(ft_d);
	cudaFree(ftx_d);
	cudaFree(fty_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(V_index_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(DxV_index);
	cudaFree(DyV_index);
	cudaFree(uVdDxV_matrix);
	cudaFree(uVdDyV_matrix);
	delete[](k);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
}


void Wigner_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP, double momk)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 1.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}

	cuDoubleComplex test= make_cuDoubleComplex(0.0, 0.0);

	double   *k = new double[N];
	for (int i = 0; i < NX; i++)
	{
		k[i] = (i-N/2) *2.0*M_PI /1.0 / LATTICE_SIZE;
	}

	// Allocate arrays on the device
	double *k_d ,*x_d, *y_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d, *V_index_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&V_index_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	//dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix, *DxV_index, *DyV_index;

	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&DyV_index, sizeof(cuDoubleComplex)  * N*N);
	cuDoubleComplex *uVdDxV_matrix, *uVdDyV_matrix;
	cudaMalloc((void**)&uVdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&uVdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix, *VdDxV_index, *VdDyV_index;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&VdDyV_index, sizeof(cuDoubleComplex)  * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);
	Udagger_Dfferential_U_make_unitarity <<<dimGrid, dimBlock >>> (uVdDxV_matrix, uVdDyV_matrix, VdDxV_matrix, VdDyV_matrix, N);

	cuDoubleComplex *ft_d,*ftx_d,*fty_d;
	cudaMalloc((void**)&ft_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&ftx_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&fty_d, sizeof(cuDoubleComplex)*N*N);

	
	initialize_vectors <<<dimGrid, dimBlock >>> (Integrated_d, Integrated2_d, N);

	for (int a = 0; a < Nc; a++) {
		for (int b = 0; b < Nc; b++) {
			

			//index_matrix_subunit <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);
			//index_matrix <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);

			//times e^ikx <- we assume that the k has only x direction
			index_matrix_fourier <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix,x_d,momk, a, b, N);
			index_matrix_fourier <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix, x_d, momk, a, b, N);
			index_matrix_fourier <<<dimGrid, dimBlock >>> (VdDxV_index, uVdDxV_matrix, x_d, momk, a, b, N);
			index_matrix_fourier <<<dimGrid, dimBlock >>> (VdDyV_index, uVdDyV_matrix, x_d, momk, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix,x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix,x_d, y_d, a, b, N);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
				exit(-1);
			}


			//Wigner_diagonal <<<dimGrid, dimBlock >>> (ft_d, V_index_d, x_d, y_d, h, 0, N, N, k_d);
			Wigner_diagonal <<<dimGrid, dimBlock >>> (ftx_d, DxV_index, DyV_index, x_d, y_d, h, 0, N, N);
			Wigner_diagonal <<<dimGrid, dimBlock >>> (fty_d, VdDxV_index, VdDyV_index, x_d, y_d, h, 0, N, N);
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed2: %s\n", cudaGetErrorString(err2));
				exit(-1);
			}


			//add_k2_sqare_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, fty_d, Integrated_d, Integrated2_d, k_d, N);
			add_complex_vector_index <<<dimGrid, dimBlock >>> (ftx_d, fty_d, Integrated_d, Integrated2_d, N);

		}
	}


	cudaMemcpy(integrated_resultDPk, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaMemcpy(integrated_resultDP, Integrated2_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(k_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(ft_d);
	cudaFree(ftx_d);
	cudaFree(fty_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(V_index_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(DxV_index);
	cudaFree(DyV_index);
	cudaFree(VdDxV_matrix);
	cudaFree(VdDyV_matrix);
	cudaFree(VdDxV_index);
	cudaFree(VdDyV_index);
	cudaFree(uVdDxV_matrix);
	cudaFree(uVdDyV_matrix);
	delete[](k);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
}



void GPD_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 1.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}

	cuDoubleComplex test= make_cuDoubleComplex(0.0, 0.0);


	// Allocate arrays on the device
	double *k_d ,*x_d, *y_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d, *V_index_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&V_index_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	//dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix, *DxV_index, *DyV_index;

	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&DyV_index, sizeof(cuDoubleComplex)  * N*N);
	cuDoubleComplex *uVdDxV_matrix, *uVdDyV_matrix;
	cudaMalloc((void**)&uVdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&uVdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	//Dfferential_U_short <<<dimGridS, dimBlockS >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N/8);
	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);
	//Take_Uzero <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);


	cuDoubleComplex *VdDxV_matrix, *VdDyV_matrix, *VdDxV_index, *VdDyV_index;
	cudaMalloc((void**)&VdDxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&VdDxV_index, sizeof(cuDoubleComplex)  * N*N);
	cudaMalloc((void**)&VdDyV_index, sizeof(cuDoubleComplex)  * N*N);

	//Udagger_Dfferential_U_short <<<dimGridS, dimBlockS >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N/8);
	Udagger_Dfferential_U <<<dimGrid, dimBlock >>> (VdDxV_matrix, VdDyV_matrix, V_matrix_d, h, N);
	Udagger_Dfferential_U_make_unitarity <<<dimGrid, dimBlock >>> (uVdDxV_matrix, uVdDyV_matrix, VdDxV_matrix, VdDyV_matrix, N);

	cuDoubleComplex *ft_d,*ftx_d,*fty_d;
	cudaMalloc((void**)&ft_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&ftx_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&fty_d, sizeof(cuDoubleComplex)*N*N);

	
	initialize_vectors <<<dimGrid, dimBlock >>> (Integrated_d, Integrated2_d, N);

	for (int a = 0; a < Nc; a++) {
		for (int b = 0; b < Nc; b++) {
			

			//index_matrix_subunit <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);
			//index_matrix <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, a, b, N);

			index_matrix <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (VdDxV_index, uVdDxV_matrix, a, b, N);
			index_matrix <<<dimGrid, dimBlock >>> (VdDyV_index, uVdDyV_matrix, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (V_index_d, V_matrix_d, x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DxV_index, DxV_matrix,x_d, y_d, a, b, N);
			//index_matrix_Gaussian <<<dimGrid, dimBlock >>> (DyV_index, DyV_matrix,x_d, y_d, a, b, N);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
				exit(-1);
			}


			add_abs_complex_vector <<<dimGrid, dimBlock >>> (DxV_index, DyV_index, Integrated_d, N);
			add_abs_complex_vector <<<dimGrid, dimBlock >>> (VdDxV_index, VdDyV_index, Integrated2_d, N);
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed2: %s\n", cudaGetErrorString(err2));
				exit(-1);
			}

		}
	}


	cudaMemcpy(integrated_resultDPk, Integrated_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	cudaMemcpy(integrated_resultDP, Integrated2_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	cudaFree(k_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(ft_d);
	cudaFree(ftx_d);
	cudaFree(fty_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(V_index_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	cudaFree(DxV_index);
	cudaFree(DyV_index);
	cudaFree(VdDxV_matrix);
	cudaFree(VdDyV_matrix);
	cudaFree(VdDxV_index);
	cudaFree(VdDyV_index);
	cudaFree(uVdDxV_matrix);
	cudaFree(uVdDyV_matrix);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
}


__global__ void UdU_complex_vector(cuDoubleComplex* V_matrix, cuDoubleComplex* V_index,int b_position, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		int recentered_index = b_position - (N*(N / 2) + N / 2);
		int m = j;
		int n = i;
		V_index[index] = make_cuDoubleComplex(0.0, 0.0);

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

		V_index[index] = trV_V;
	}
}


__global__ void DUdDU_complex_vector(cuDoubleComplex* dVx_matrix, cuDoubleComplex* dVy_matrix, cuDoubleComplex* V_index, int b_position, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		int recentered_index = b_position - (N*(N / 2) + N / 2);
		int m = j;
		int n = i;
		V_index[index] = make_cuDoubleComplex(0.0, 0.0);

		cuDoubleComplex trV_V = make_cuDoubleComplex(0.0, 0.0);
		//V(out of the region)=1 -> tr(V^dagger V) = sum_i sum_j V^*_{i j} V_{i j} = 3
		if (((recentered_index + m * N + n) < 0 || (recentered_index + m * N + n) > (N)*(N)-1)
			&& ((recentered_index + (N - m) * N + N - n) >= 0 && (recentered_index + (N - m) * N + N - n) <= (N)*(N)-1)) {
			//trV_V = make_cuDoubleComplex(3.0, 0.0);
			for (int tr = 0; tr < 3; ++tr) {
				trV_V = make_cuDoubleComplex(0.0, 0.0);
			}
		}
		else if (((recentered_index + m * N + n) >= 0 && (recentered_index + m * N + n) <= (N)*(N)-1)
			&& ((recentered_index + (N - m) * N + N - n) < 0 || (recentered_index + (N - m) * N + N - n) > (N)*(N)-1)) {

			for (int tr = 0; tr < 3; ++tr) {
				trV_V = make_cuDoubleComplex(0.0, 0.0);
			}
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
						cuCmul(cuConj(dVx_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), dVx_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
					trV_V = cuCadd(trV_V,
						cuCmul(cuConj(dVy_matrix[3 * 3 * (recentered_index + m * N + n) + 3 * in + tr]), dVy_matrix[3 * 3 * (recentered_index + (N - m) * N + N - n) + 3 * in + tr]));
				}
			}
		}

		V_index[index] = trV_V;
	}
}


__global__ void times_k2_sqare_times_h_complex_vector(cuDoubleComplex *ft, 
	cuDoubleComplex *Integrated, double *k_d,double h, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		Integrated[index] = make_cuDoubleComplex(0.0,0.0);
		double simpson1 = 1.0;
		double simpson2 = 1.0;
		int diffinitm = j;
		if (j == 0 || j == N - 1) {
			simpson1 = 1.0 / 3.0;
		}
		else if (diffinitm % 2 == 0) {
			simpson1 = 2.0 / 3.0;
		}
		else {

			simpson1 = 4.0 / 3.0;
		}


		if (i == 0 || i == N - 1) {
			simpson2 = 1.0 / 3.0;
		}
		else if (i % 2 == 0) {
			simpson2 = 2.0 / 3.0;
		}
		else {

			simpson2 = 4.0 / 3.0;
		}

		//cuDoubleComplex k2_comp = make_cuDoubleComplex(1.0, 0.0);

		if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {

			cuDoubleComplex k2_comp = make_cuDoubleComplex(simpson1*simpson2*(k_d[j] * k_d[j] + k_d[i] * k_d[i])
				*h*h*(double(LATTICE_SIZE))/(double(NX))*(double(LATTICE_SIZE)) / (double(NX)),
				simpson1*simpson2*(k_d[j] * k_d[j] + k_d[i] * k_d[i])
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)));
			Integrated[index] =  cuCmul(k2_comp, ft[index]);
		}
		else {
			cuDoubleComplex k2_comp = make_cuDoubleComplex(-simpson1*simpson2*(k_d[j] * k_d[j] + k_d[i] * k_d[i])
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)),
				-simpson1*simpson2*(k_d[j] * k_d[j] + k_d[i] * k_d[i])
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)));
			Integrated[index] =  cuCmul(k2_comp,  ft[index]);
		}

	}
}


__global__ void times_k2_sqare_times_h_for_DU_complex_vector(cuDoubleComplex *ft,
	cuDoubleComplex *Integrated, double *k_d, double h, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		Integrated[index] = make_cuDoubleComplex(0.0, 0.0);
		double simpson1 = 1.0;
		double simpson2 = 1.0;
		int diffinitm = j;
		if (j == 0 || j == N - 1) {
			simpson1 = 1.0 / 3.0;
		}
		else if (diffinitm % 2 == 0) {
			simpson1 = 2.0 / 3.0;
		}
		else {

			simpson1 = 4.0 / 3.0;
		}


		if (i == 0 || i == N - 1) {
			simpson2 = 1.0 / 3.0;
		}
		else if (i % 2 == 0) {
			simpson2 = 2.0 / 3.0;
		}
		else {

			simpson2 = 4.0 / 3.0;
		}

		//cuDoubleComplex k2_comp = make_cuDoubleComplex(1.0, 0.0);

		if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {

			cuDoubleComplex k2_comp = make_cuDoubleComplex(simpson1*simpson2
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)),
				simpson1*simpson2
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)));
			Integrated[index] = cuCmul(k2_comp, ft[index]);
		}
		else {
			cuDoubleComplex k2_comp = make_cuDoubleComplex(-simpson1*simpson2
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)),
				-simpson1*simpson2
				*h*h*(double(LATTICE_SIZE)) / (double(NX))*(double(LATTICE_SIZE)) / (double(NX)));
			Integrated[index] = cuCmul(k2_comp, ft[index]);
		}

	}
}



// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

//This code is taken from https://qiita.com/gyu-don/items/ef8a128fa24f6bddd342
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n)
{
	// shared memoryは、うまく使えば速度が速い。(ここでは、うまく使ってない。後で直す)
	// 同じブロック内のスレッド間で共有される。
	//extern __shared__ int sdata[];

	T *sdata = SharedMemory<T>();

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = (i < n) ? g_idata[i] : 0;
	__syncthreads();

	// 例えばtidが0..7の場合、
	// sdata[0] += sdata[1]; sdata[2] += sdata[3]; sdata[4] += sdata[5]; sdata[6] += sdata[7];
	// sdata[0] += sdata[2]; sdata[4] += sdata[6];
	// sdata[0] += sdata[4];
	// のように、sdata[0]に集められる。
	// do reduction in shared mem
	for (unsigned int s = 1; s<blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kepler以降では、ワープ内の別スレッドから変数を読む、shuffle operationというものがある。
// 1ワープに収まって以降は、これを使えば速くなる。
// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/ によると、
// sharedメモリを使うよりも命令数が少なくて済むから速いらしい。
// shuffle operationが使えないアーキテクチャではループアンロールでもしとけ、とNVIDIAのサンプルにはある。
// ちょっと古めのNVIDIAの資料には、同じワープ内だとどうせ全部一緒に動くから__syncthreads不要だよ、とあったが、
// サンプルでは、__syncthreadsされている。このへん、何か変わったんだろうか?
// あと、__shfl_downしてるところではループアンロールしなくていいのかなぁ。

template <class T>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int n)
{
	//extern __shared__ int sdata[];
	T *sdata = SharedMemory<T>();

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	T mysum = (i < n) ? g_idata[i] : 0;
	if (i + blockDim.x < n) mysum += g_idata[i + blockDim.x];
	sdata[tid] = mysum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1) {
		if (tid < s) {
			sdata[tid] = mysum = mysum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if (blockDim.x >= 64) mysum += sdata[tid + 32];
		for (int offset = 32 / 2; offset>0; offset >>= 1) {
			mysum += __shfl_down(mysum, offset);
		}
	}
	if (tid == 0) g_odata[blockIdx.x] = mysum;
#else
#error "__shfl_down requires CUDA arch >= 300."
#endif
}

__global__ void simple_definition_vector(int *test_vector,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		test_vector[index] = index;
	}
}

void test_reduction()
{
	int N = 4096;
	std::vector<double> test_int(N*N, 0);
	printf("N*N:      %10d\n", N*N);
	double cpu_result = 0;
	for (int i = 0; i<N*N; i++) test_int[i]=double(i*0.1);
	clock_t cl = clock();
	//need #include <chrono>
	//auto cpu_start = std::chrono::system_clock::now();
	for (int i = 0; i<N*N; i++) cpu_result += test_int[i];
	//printf("%d\n", cpu_result);
	std::cout << cpu_result << '\n';
	std::cout << (double)(N*N*(N*N-1.0)/2.0*0.1) << '\n';
	//assert(cpu_result == N*N);
	clock_t cl_end = clock();
	double time = (double)(cl_end - cl) / (double)CLOCKS_PER_SEC * 1000.0;
	//auto cpu_end = std::chrono::system_clock::now();       // 計測終了時刻を保存
	//auto cpu_dur = cpu_end - cpu_start;        // 要した時間を計算
	//auto time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_dur).count();
	printf("CPU:    %10.4f\n", time);


	double *arr_dev;
	cudaMalloc((void**)&arr_dev, sizeof(double) * N*N);
	cudaMemcpy(arr_dev, test_int.data(), N*N * sizeof(double), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	int th = 16;
	int blocks = (N*N - 1) / th + 1;
	int shared_mem_size = 2 * th * sizeof(int);
	double *out1_dev, *out2_dev;
	cudaMalloc((void**)&out1_dev, sizeof(double) * blocks);
	cudaMalloc((void**)&out2_dev, sizeof(double) * blocks);

	cudaEventRecord(start);
	double **in = &arr_dev, **out = &out1_dev;
	int n = N*N;
	while (blocks > 1) {
		reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
		//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			puts(cudaGetErrorString(err));
		}
		if (*out == out1_dev) {
			out = &out2_dev; in = &out1_dev;
		}
		else {
			out = &out1_dev; in = &out2_dev;
		}
		n = blocks;
		blocks = (blocks - 1) / th + 1;
		cudaThreadSynchronize();
	}

	reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
	//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
	double result;
	cudaMemcpy(&result, *out, sizeof(double), cudaMemcpyDeviceToHost);
	//printf("%d\n", result);
	std::cout << result << '\n';
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//assert(result == N*N);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%8.4f  ", milliseconds);

	cudaFree(out1_dev);
	cudaFree(out2_dev);
	printf("\n");


	cudaFree(arr_dev);
}


__global__ void take_real_part_vector(cuDoubleComplex *test_vector,double *real_vector, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		real_vector[index] = test_vector[index].x;
	}
}


void GPD_FFT(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 1.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
		s = 0.1, s2 = s*s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *u_a = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			//f[N*j + i] = ;
		}
	}

	cuDoubleComplex test = make_cuDoubleComplex(0.0, 0.0);

	double   *k = new double[N];
	for (int i = 0; i <= N / 2; i++)
	{
		k[i] = double(i) * 2.0 * M_PI / double(LATTICE_SIZE);
	}
	for (int i = N / 2 + 1; i < N; i++)
	{
		k[i] = double(i - N) * 2.0 * M_PI / double(LATTICE_SIZE);
	}
	double h_k = 2.0 * M_PI / double(LATTICE_SIZE);

	// Allocate arrays on the device
	double *k_d, *x_d, *y_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&x_d, sizeof(double)*N*N);
	cudaMalloc((void**)&y_d, sizeof(double)*N*N);
	cudaMemcpy(x_d, x, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(double)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);


	cuDoubleComplex *V_matrix_d;
	cudaMalloc((void**)&V_matrix_d, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMemcpy(V_matrix_d, V_matrix, sizeof(std::complex<double>) * 3 * 3 * N*N, cudaMemcpyHostToDevice);

	cuDoubleComplex *Integrated_d, *Integrated2_d, *V_index_d;
	cudaMalloc((void**)&Integrated_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&Integrated2_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&V_index_d, sizeof(cuDoubleComplex)*N*N);

	//dim3 dimGridS(int((N / 8 - 0.5) / BSZ) + 1, int((N / 8 - 0.5) / BSZ) + 1);
	//dim3 dimBlockS(BSZ, BSZ);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	cuDoubleComplex *DxV_matrix, *DyV_matrix;

	cudaMalloc((void**)&DxV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);
	cudaMalloc((void**)&DyV_matrix, sizeof(cuDoubleComplex) * 3 * 3 * N*N);

	Dfferential_U <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_matrix_d, h, N);

	cuDoubleComplex *ft_d, *ftx_d, *fty_d;
	cudaMalloc((void**)&ft_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&ftx_d, sizeof(cuDoubleComplex)*N*N);
	cudaMalloc((void**)&fty_d, sizeof(cuDoubleComplex)*N*N);


	initialize_vectors <<<dimGrid, dimBlock >>> (Integrated_d, Integrated2_d, N);

	for (int a = 0; a < NX; a++) {
		int b_space = a*NX + a;

		//UdU_complex_vector <<<dimGrid, dimBlock >>> (V_matrix_d, V_index_d, b_space, N);
		DUdDU_complex_vector <<<dimGrid, dimBlock >>> (DxV_matrix, DyV_matrix, V_index_d, b_space, N);


		std::complex<double>  *V_index= new std::complex<double>[N*N];
		cudaMemcpy(V_index, V_index_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
			std::cout << "close the window" << '\n';
			exit(-1);
		}

		cufftHandle plan;
		cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

		cufftExecZ2Z(plan, V_index_d, ft_d, CUFFT_FORWARD);



		//times k^2 * h^2
		//times_k2_sqare_times_h_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, k_d, h_k, N);
		times_k2_sqare_times_h_for_DU_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, k_d, h_k, N);
		cudaError_t err2 = cudaGetLastError();
		if (err2 != cudaSuccess) {
			fprintf(stderr, "kernel launch failed2: %s\n", cudaGetErrorString(err2));
			std::cout << "close the window" << '\n';
			exit(-1);
		}

		double *real_ftx_d;
		cudaMalloc((void**)&real_ftx_d, sizeof(double)*N*N);

		take_real_part_vector <<<dimGrid, dimBlock >>> (ftx_d, real_ftx_d, N);

		int th = 16;
		int blocks = (N*N - 1) / th + 1;
		int shared_mem_size = 2 * th * sizeof(int);
		double *out1_dev, *out2_dev;
		cudaMalloc((void**)&out1_dev, sizeof(double) * blocks);
		cudaMalloc((void**)&out2_dev, sizeof(double) * blocks);

		double **in = &real_ftx_d, **out = &out1_dev;
		int n = N*N;
		while (blocks > 1) {
			reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
			//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				puts(cudaGetErrorString(err));
			}
			if (*out == out1_dev) {
				out = &out2_dev; in = &out1_dev;
			}
			else {
				out = &out1_dev; in = &out2_dev;
			}
			n = blocks;
			blocks = (blocks - 1) / th + 1;
			cudaThreadSynchronize();
		}

		cudaError_t err3 = cudaGetLastError();
		if (err3 != cudaSuccess) {
			fprintf(stderr, "kernel launch failed3: %s\n", cudaGetErrorString(err3));
			std::cout << "close the window" << '\n';
			exit(-1);
		}
		reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
		//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
		double result;
		cudaMemcpy(&result, *out, sizeof(double), cudaMemcpyDeviceToHost);
		integrated_resultDPk[b_space] = std::complex<double>(result, 0);
		cudaError_t err4 = cudaGetLastError();
		if (err4 != cudaSuccess) {
			fprintf(stderr, "kernel launch failed4: %s\n", cudaGetErrorString(err4));
			std::cout << "close the window" << '\n';
			exit(-1);
		}

		cudaFree(out1_dev);
		cudaFree(out2_dev);
		cudaFree(real_ftx_d);
		cufftDestroy(plan);

		delete[]V_index;
	}
	
	for (int a = 0; a < NX; a++) {
		if (a != N / 2) {
			int b_space = a*N + N - a;

			//UdU_complex_vector <<<dimGrid, dimBlock >>> (V_matrix_d, V_index_d, b_space, N);
			DUdDU_complex_vector << <dimGrid, dimBlock >> > (DxV_matrix, DyV_matrix, V_index_d, b_space, N);

			cudaError_t err3 = cudaGetLastError();
			if (err3 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed first: %s\n", cudaGetErrorString(err3));
				std::cout << "close the window" << '\n';
				exit(-1);
			}

			std::complex<double>  *V_index = new std::complex<double>[N*N];
			cudaMemcpy(V_index, V_index_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
				std::cout << "close the window" << '\n';
				exit(-1);
			}

			cufftHandle plan;
			cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

			cufftExecZ2Z(plan, V_index_d, ft_d, CUFFT_FORWARD);


			//times_k2_sqare_times_h_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, k_d, h_k, N);
			times_k2_sqare_times_h_for_DU_complex_vector <<<dimGrid, dimBlock >>> (ft_d, ftx_d, k_d, h_k, N);
			cudaError_t err2 = cudaGetLastError();
			if (err2 != cudaSuccess) {
				fprintf(stderr, "kernel launch failed2: %s\n", cudaGetErrorString(err2));
				exit(-1);
			}

			double *real_ftx_d;
			cudaMalloc((void**)&real_ftx_d, sizeof(double)*N*N);

			take_real_part_vector <<<dimGrid, dimBlock >>> (ftx_d, real_ftx_d, N);

			int th = 16;
			int blocks = (N*N - 1) / th + 1;
			int shared_mem_size = 2 * th * sizeof(int);
			double *out1_dev, *out2_dev;
			cudaMalloc((void**)&out1_dev, sizeof(double) * blocks);
			cudaMalloc((void**)&out2_dev, sizeof(double) * blocks);

			double **in = &real_ftx_d, **out = &out1_dev;
			int n = N*N;
			while (blocks > 1) {
				reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
				//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
				cudaError_t err = cudaGetLastError();
				if (err != cudaSuccess) {
					puts(cudaGetErrorString(err));
				}
				if (*out == out1_dev) {
					out = &out2_dev; in = &out1_dev;
				}
				else {
					out = &out1_dev; in = &out2_dev;
				}
				n = blocks;
				blocks = (blocks - 1) / th + 1;
				cudaThreadSynchronize();
			}

			reduce0<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
			//reduce4<double> <<<blocks, th, shared_mem_size >>> (*in, *out, n);
			double result;
			cudaMemcpy(&result, *out, sizeof(double), cudaMemcpyDeviceToHost);
			integrated_resultDPk[b_space] = std::complex<double>(result,0.0);


			cudaFree(out1_dev);
			cudaFree(out2_dev);
			cudaFree(real_ftx_d);
			cufftDestroy(plan);

			delete[]V_index;
		}
	}

	cudaFree(k_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(ft_d);
	cudaFree(ftx_d);
	cudaFree(fty_d);
	cudaFree(Integrated_d);
	cudaFree(Integrated2_d);
	cudaFree(V_matrix_d);
	cudaFree(V_index_d);
	cudaFree(DxV_matrix);
	cudaFree(DyV_matrix);
	delete[](k);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
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


