//lattice size are 8 times 8 femtometer 
#include <cufft.h> 
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
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

#include"Header_Params.h"


//parameters for solving the JIMWLK equation
extern const double Rp ;
extern const double R_CQ;
extern const double g2_mu_Rp ;
extern const double m_Rp ;
extern const double mass;
//1.0 makes the LATTICE_SIZE and NX cast double
extern const double lattice_spacing ;


__global__ void solve_poisson_with_mass(cufftDoubleComplex *ft, cufftDoubleComplex *ft_k, double *k, double *mass_d, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		double k2 = k[i] * k[i] + k[j] * k[j] + mass_d[0] * mass_d[0];
		if (k2 < 1.0e-30) { k2 = 1.0; }
		ft_k[index].x = -ft[index].x *1.0 / k2;
		ft_k[index].y = -ft[index].y *1.0 / k2;
	}
}

__global__ void solve_poisson_with_mass_and_v(cufftDoubleComplex *ft, cufftDoubleComplex *ft_k, double *k, double *mass_d, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		double k2 = k[i] * k[i] + k[j] * k[j] + mass_d[0] * mass_d[0];
#ifdef v_Parameter
		double v_factor = exp(-v_Parameter*sqrt(k[i] * k[i] + k[j] * k[j]));
#else
		double v_factor = 1.0;
#endif
		if (k2 < 1.0e-30) { k2 = 1.0; }
		ft_k[index].x = -ft[index].x *1.0 / k2*v_factor;
		ft_k[index].y = -ft[index].y *1.0 / k2*v_factor;
	}
}

//__global__ void solve_poisson_with_mass(cufftDoubleComplex *ft, cufftDoubleComplex *ft_k, double *k,double *mass_d, int N)
//{
//	int i = threadIdx.x + blockIdx.x*BSZ;
//	int j = threadIdx.y + blockIdx.y*BSZ;
//	int index = j*N + i;
//	if (i<N && j<N)
//	{
//		double k2 = k[i] * k[i] + k[j] * k[j] + mass_d[0]*mass_d[0];
//		if (i == 0 && j == 0) { k2 = 1.0; }
//		ft_k[index].x = -ft[index].x / k2;
//		ft_k[index].y = -ft[index].y / k2;
//	}
//}


__global__ void convolve_f_and_g(cufftDoubleComplex *ft, cufftDoubleComplex *gt, cufftDoubleComplex *cvt_k, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
		if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)){
			cvt_k[index].x = (ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = (ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
		else {

			cvt_k[index].x = -(ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = -(ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
	}
}



__global__ void convolve_f_and_g_with_vParam(cufftDoubleComplex *ft, cufftDoubleComplex *gt, cufftDoubleComplex *cvt_k,double *k, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j * N + i;
	if (i<N && j<N)
	{
#ifdef v_Parameter
		double v_factor = exp(-v_Parameter*sqrt(k[i] * k[i] + k[j] * k[j]));
#else
		double v_factor = 1.0;
#endif

		if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
			cvt_k[index].x = v_factor*(ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = v_factor*(ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
		else {

			cvt_k[index].x = -v_factor*(ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = -v_factor*(ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
	}
}


__global__ void convolve_1D_f_and_g(cufftDoubleComplex *ft, cufftDoubleComplex *gt, cufftDoubleComplex *cvt_k, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int index = i;
	if (i<N)
	{
		if (i % 2 == 0) {
			cvt_k[index].x = (ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = (ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
		else {

			cvt_k[index].x = -(ft[index].x * gt[index].x *1.0 - ft[index].y * gt[index].y *1.0);
			cvt_k[index].y = -(ft[index].y * gt[index].x *1.0 + ft[index].x * gt[index].y *1.0);
		}
	}
}


__global__ void real2complex(double *f, cufftDoubleComplex *fc, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fc[index].x = f[index];
		fc[index].y = 0.0;
	}
}


//__global__ void real2complex(double *f, cufftDoubleComplex *fc, int N)
//{
//	int i = threadIdx.x + blockIdx.x*BSZ;
//	int j = threadIdx.y + blockIdx.y*BSZ;
//	int index = j*N + i;
//	if (i<N && j<N)
//	{
//		fc[index].x = f[index];
//		fc[index].y = 0.0;
//	}
//}

__global__ void complex2real(cufftDoubleComplex *fc, double *f, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		f[index] = fc[index].x *1.0/ ((double)(N*N));
		//divide by number of elements to recover value
	}
}

//__global__ void complex2real(cufftDoubleComplex *fc, double *f, int N)
//{
//	int i = threadIdx.x + blockIdx.x*BSZ;
//	int j = threadIdx.y + blockIdx.y*BSZ;
//	int index = j*N + i;
//	if (i<N && j<N)
//	{
//		f[index] = fc[index].x / ((double)(N*N));
		//divide by number of elements to recover value
//	}
//}


void Solve_Poisson_Equation(double* rho,double* Solution)
{

	int N = NX;
	double h = lattice_spacing;
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
			r2 = (x[N*j + i] )*(x[N*j + i]) + (y[N*j + i] )*(y[N*j + i] );
			//f[N*j + i] = (r2 - 2 * s2 - mass*mass*s2*s2) / (s2*s2)*exp(-r2 / (2 * s2));
			f[N*j + i] = rho[N*j+i];

		}
	}

	double   *k = new double[N];
	for (int i = 0; i <= N / 2; i++)
	{
		k[i] = i * 2.0 * M_PI/ LATTICE_SIZE;
	}
	for (int i = N / 2 + 1; i < N; i++)
	{
		k[i] = (i - N) * 2.0 * M_PI/ LATTICE_SIZE;
	}

	std::complex<double>* f_Comp = new std::complex<double>[N*N];

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			f_Comp[N*j + i] = std::complex<double>(f[N*j + i],0.0);
		}
	}



	//for (int j = 0; j < N; j++) {
	//	for (int i = 0; i < N; i++)
	//	{
	//		f[N*j + i] = f_Comp[N*j + i].real()/((double)(N*N));
	//	}
	//}


	// Allocate arrays on the device
	double *k_d, *f_d, *u_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMalloc((void**)&f_d, sizeof(double)*N*N);
	cudaMalloc((void**)&u_d, sizeof(double)*N*N);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(f_d, f, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	double *mass_d;
	cudaMalloc((void**)&mass_d, sizeof(double));
	cudaMemcpy(mass_d, &mass, sizeof(double), cudaMemcpyHostToDevice);

	//double mass_r;
	//cudaMemcpy(&mass_r, mass_d, sizeof(double), cudaMemcpyDeviceToHost);

	//std::cout << mass_r[0] << std::endl;


	cufftDoubleComplex *ft_d, *f_dc, *ft_d_k, *u_dc;
	cudaMalloc((void**)&ft_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&ft_d_k, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&u_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMemcpy(f_dc, f_Comp, sizeof(std::complex<double>)*N*N, cudaMemcpyHostToDevice);


	//cudaMemcpy(f_Comp, f_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	//real2complex <<<dimGrid, dimBlock >>> (f_d, f_dc, N);

	//complex2real <<<dimGrid, dimBlock >>> (f_dc, f_d, N);
	//cudaMemcpy(fs, f_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);



	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

	cufftExecZ2Z(plan, f_dc, ft_d, CUFFT_FORWARD);

	//cudaMemcpy(f_Comp, ft_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	//complex2real <<<dimGrid, dimBlock >>> (ft_d, f_d, N);
	//cudaMemcpy(fs, f_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
#ifdef v_Parameter
	solve_poisson_with_mass_and_v <<<dimGrid, dimBlock >>> (ft_d, ft_d_k, k_d,mass_d, N);
#else
	solve_poisson_with_mass <<<dimGrid, dimBlock >>> (ft_d, ft_d_k, k_d,mass_d, N);
#endif
	
	//complex2real <<<dimGrid, dimBlock >>> (ft_d, f_d, N);
	//cudaMemcpy(fs, f_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);


	cufftExecZ2Z(plan, ft_d_k, u_dc, CUFFT_INVERSE);
	//complex2real <<<dimGrid, dimBlock >>> (u_dc, u_d, N);


	cudaMemcpy(f_Comp, u_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			Solution[N*j + i] = f_Comp[N*j + i].real() / ((double)(N*N));
		}
	}


	//cudaMemcpy(Solution, u_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	double constant = Solution[0];
	for (int i = 0; i < N*N; i++)
	{
	//	Solution[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	}

#ifdef OUTPUT_FT_NOISE
	std::ostringstream ofilename_i;
	ofilename_i << "E:\\hagiyoshi\\Data\\JIMWLK\\solution_Poisson_cpp.txt";
	//ofilename_i << "solution_Poisson_cpp.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t"  << "\n";

	//std::cout << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t"  << "\n";

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i] << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}
#endif

	cudaFree(k_d);
	cudaFree(f_d);
	cudaFree(u_d);
	cudaFree(ft_d);
	cudaFree(f_dc);
	cudaFree(ft_d_k);
	cudaFree(u_dc);
	cudaFree(mass_d);
	cufftDestroy(plan);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](u_a);
	delete[](err);
	delete[](k);
	delete[]f_Comp;

	//return mass_r;
}


void Calculate_Convolution(double* func1,double* func2, double* Convolution)
{

	int N = NX;
	double h = lattice_spacing;
	double   xmax = h * N / 2.0, xmin = -h * N / 2.0, ymin = -h * N / 2.0,
		s = 0.1, s2 = s * s;
	double   *x = new double[N*N], *y = new double[N*N],
		*f = new double[N*N], *g = new double[N*N], *u_a = new double[N*N], *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i * h;
			y[N*j + i] = ymin + j * h;
			//r2 = (x[N*j + i])*(x[N*j + i]) + (y[N*j + i])*(y[N*j + i]);
			//f[N*j + i] = (r2 - 2 * s2 - mass*mass*s2*s2) / (s2*s2)*exp(-r2 / (2 * s2));
			f[N*j + i] = func1[N*j + i];
			g[N*j + i] = func2[N*j + i];

		}
	}

	std::complex<double>* f_Comp = new std::complex<double>[N*N];
	std::complex<double>* g_Comp = new std::complex<double>[N*N];

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			f_Comp[N*j + i] = std::complex<double>(f[N*j + i], 0.0);
			g_Comp[N*j + i] = std::complex<double>(g[N*j + i], 0.0);
		}
	}
	

	// Allocate arrays on the device
	double  *f_d, *u_d, *us_d;
	cudaMalloc((void**)&f_d, sizeof(double)*N*N);
	cudaMalloc((void**)&u_d, sizeof(double)*N*N);
	cudaMalloc((void**)&us_d, sizeof(double)*N*N);
	cudaMemcpy(f_d, f, sizeof(double)*N*N, cudaMemcpyHostToDevice);

	cufftDoubleComplex *ft_d,*gt_d, *f_dc,*g_dc, *cvt_d, *u_dc;
	cudaMalloc((void**)&ft_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&gt_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&cvt_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&g_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&u_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMemcpy(f_dc, f_Comp, sizeof(std::complex<double>)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(g_dc, g_Comp, sizeof(std::complex<double>)*N*N, cudaMemcpyHostToDevice);


	//cudaMemcpy(f_Comp, f_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);

	double* fs = new double[N*N];



	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

	cufftExecZ2Z(plan, f_dc, ft_d, CUFFT_FORWARD);
	cufftExecZ2Z(plan, g_dc, gt_d, CUFFT_FORWARD);


	//cudaMemcpy(f_Comp, ft_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);



	convolve_f_and_g <<<dimGrid, dimBlock >>> (ft_d, gt_d, cvt_d, N);


	cufftExecZ2Z(plan, cvt_d, u_dc, CUFFT_INVERSE);


	cudaMemcpy(f_Comp, u_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			Convolution[N*j + i] = f_Comp[N*j + i].real() * 1.0*LATTICE_SIZE*1.0*LATTICE_SIZE / ((double)(N*N)) / ((double)(N*N));
		}
	}


	//cudaMemcpy(Convolution, u_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	//double constant = Convolution[0];
	//for (int i = 0; i < N*N; i++)
	//{
	//	Convolution[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	//}

#ifdef OUTPUT_CONVOLUTION
	std::ostringstream ofilename_i;
	ofilename_i << "E:\\hagiyoshi\\Data\\JIMWLK\\Convolution.txt";
	//ofilename_i << "Convolution.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t" << "\n";

	//std::cout << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t"  << "\n";

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Convolution[N*j + i] << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Convolution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}
#endif

	cudaFree(f_d);
	cudaFree(u_d);
	cudaFree(ft_d);
	cudaFree(f_dc);
	cudaFree(g_dc);
	cudaFree(cvt_d);
	cudaFree(u_dc);
	cufftDestroy(plan);
	delete[](x);
	delete[](y);
	delete[](f);
	delete[](g);
	delete[](u_a);
	delete[](err);
	delete[]f_Comp;
	delete[]g_Comp;

}


void Calculate_Convolution_complex(std::complex<double>* func1, std::complex<double>* func2)
{

	int N = NX;
	double h = lattice_spacing;
	double   xmax = h * N / 2.0, xmin = -h * N / 2.0, ymin = -h * N / 2.0,
		s = 0.1, s2 = s * s;
	double   *x = new double[N*N], *y = new double[N*N],
		  *err = new double[N*N];
	double r2;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i * h;
			y[N*j + i] = ymin + j * h;

		}
	}

	double   *k = new double[N];
	for (int i = 0; i <= N / 2; i++)
	{
		k[i] = i * 2.0 * M_PI / LATTICE_SIZE;
	}
	for (int i = N / 2 + 1; i < N; i++)
	{
		k[i] = (i - N) * 2.0 * M_PI / LATTICE_SIZE;
	}


	double *k_d;
	cudaMalloc((void**)&k_d, sizeof(double)*N);
	cudaMemcpy(k_d, k, sizeof(double)*N, cudaMemcpyHostToDevice);

	// Allocate arrays on the device
	cufftDoubleComplex *ft_d, *gt_d, *f_dc, *g_dc, *cvt_d, *u_dc;
	cudaMalloc((void**)&ft_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&gt_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&cvt_d, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&g_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMalloc((void**)&u_dc, sizeof(cufftDoubleComplex)*N*N);
	cudaMemcpy(f_dc, func1, sizeof(std::complex<double>)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(g_dc, func2, sizeof(std::complex<double>)*N*N, cudaMemcpyHostToDevice);


	//cudaMemcpy(f_Comp, f_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);



	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

	cufftExecZ2Z(plan, f_dc, ft_d, CUFFT_FORWARD);
	cufftExecZ2Z(plan, g_dc, gt_d, CUFFT_FORWARD);


	//cudaMemcpy(f_Comp, ft_d, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);

#ifdef v_Parameter
	//convolve_f_and_g_with_vParam <<<dimGrid, dimBlock >>> (ft_d, gt_d, cvt_d,k_d, N);
	
	convolve_f_and_g <<<dimGrid, dimBlock >>> (ft_d, gt_d, cvt_d, N);
#else
	convolve_f_and_g <<<dimGrid, dimBlock >>> (ft_d, gt_d, cvt_d, N);
#endif


	cufftExecZ2Z(plan, cvt_d, u_dc, CUFFT_INVERSE);


	cudaMemcpy(func2, u_dc, sizeof(std::complex<double>)*N*N, cudaMemcpyDeviceToHost);


	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			std::complex<double> factor(1.0*LATTICE_SIZE*1.0*LATTICE_SIZE / ((double)(N*N)) / ((double)(N*N)), 0.0);
			func2[N*j + i] = func2[N*j + i] * factor;
		}
	}


	//cudaMemcpy(Convolution, u_d, sizeof(double)*N*N, cudaMemcpyDeviceToHost);

	//double constant = Convolution[0];
	//for (int i = 0; i < N*N; i++)
	//{
	//	Convolution[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	//}

#ifdef OUTPUT_CONVOLUTION
	std::ostringstream ofilename_i;
	ofilename_i << "G:\\hagiyoshi\\Data\\test_FFT\\ConvolutionComp.txt";
	//ofilename_i << "Convolution.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t" << "\n";

	//std::cout << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t"  << "\n";

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Convolution[N*j + i].real() << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Convolution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}
#endif

	cudaFree(k_d);
	cudaFree(ft_d);
	cudaFree(gt_d);
	cudaFree(f_dc);
	cudaFree(g_dc);
	cudaFree(cvt_d);
	cudaFree(u_dc);
	cufftDestroy(plan);
	delete[](x);
	delete[](k);
	delete[](y);
	delete[](err);

}


void Calculate_Convolution_1D()
{

	int N = NX;
	double h = lattice_spacing;
	double   xmax = h * N / 2.0, xmin = -h * N / 2.0;
	double   *x = new double[N],
		*f = new double[N], *g = new double[N], *err = new double[N],
		*Convolution = new double[N], *Convolution_analytic = new double[N];
	std::vector<double> FT_gauss(N, 0);
	double r2;
	for (int i = 0; i< N; i++) {
		x[i] = xmin + i * h;
		f[i] = x[i] * x[i] * x[i] * x[i] * x[i] * exp(-x[i] * x[i]);
		g[i] = exp(-x[i] * x[i] * 4.0);

		Convolution_analytic[i] = 1.0 / 3125.0*sqrt(M_PI / 5.0)
			*x[i] * (1024.0*x[i] * x[i] * x[i] * x[i] + 1600.0*x[i] * x[i] + 375.0)
			*exp(-4.0 / 5.0*(x[i] * x[i]));
		FT_gauss[i] = sqrt(M_PI) / 2.0*exp(-x[i] * x[i] / 16.0);
	}

	std::complex<double>* f_Comp = new std::complex<double>[N];
	std::complex<double>* g_Comp = new std::complex<double>[N];

	for (int i = 0; i< N; i++) {
		f_Comp[i] = std::complex<double>(f[i], 0.0);
		g_Comp[i] = std::complex<double>(g[i], 0.0);
	}

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
	cufftDoubleComplex *ft_d, *gt_d, *f_dc, *g_dc, *cvt_d, *u_dc;
	cudaMalloc((void**)&ft_d, sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&gt_d, sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&cvt_d, sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&g_dc, sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&u_dc, sizeof(cufftDoubleComplex)*N);
	cudaMemcpy(f_dc, f_Comp, sizeof(std::complex<double>)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(g_dc, g_Comp, sizeof(std::complex<double>)*N, cudaMemcpyHostToDevice);


	//cudaMemcpy(f_Comp, f_dc, sizeof(std::complex<double>)*N, cudaMemcpyDeviceToHost);

	dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
	dim3 dimBlock(BSZ, BSZ);


	cufftHandle plan;
	cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);

	cufftExecZ2Z(plan, f_dc, ft_d, CUFFT_FORWARD);
	cufftExecZ2Z(plan, g_dc, gt_d, CUFFT_FORWARD);


	cudaMemcpy(g_Comp, gt_d, sizeof(std::complex<double>)*N, cudaMemcpyDeviceToHost);

	std::ostringstream ofilename_f;
	ofilename_f << "G:\\hagiyoshi\\Data\\test_FFT\\FT_1D.txt";
	//ofilename_i << "Convolution.txt";
	std::ofstream ofs_res_f(ofilename_f.str().c_str());

	ofs_res_f << "#k" << "\t" << "ft numeric" << "\t" << "ft analytic" << "\t" << "\n";

	for (int i = 0; i < N; i++) {
		if (i % 2 == 0) {

			ofs_res_f << k[i] << "\t" << g_Comp[i].real() * 1.0*LATTICE_SIZE / ((double)(N)) << "\t" << FT_gauss[i] << "\n";
		}
		else {
			ofs_res_f << k[i] << "\t" << -g_Comp[i].real() * 1.0*LATTICE_SIZE / ((double)(N)) << "\t" << FT_gauss[i] << "\n";
		}
	}


	convolve_1D_f_and_g <<<dimGrid, dimBlock >>> (ft_d, gt_d, cvt_d, N);


	cufftExecZ2Z(plan, cvt_d, u_dc, CUFFT_INVERSE);


	cudaMemcpy(f_Comp, u_dc, sizeof(std::complex<double>)*N, cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++) {
		Convolution[i] = f_Comp[i].real() * 1.0*LATTICE_SIZE / ((double)(N*N));
	}


	//cudaMemcpy(Convolution, u_d, sizeof(double)*N, cudaMemcpyDeviceToHost);

	//double constant = Convolution[0];
	//for (int i = 0; i < N; i++)
	//{
	//	Convolution[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	//}

	std::ostringstream ofilename_i;
	ofilename_i << "G:\\hagiyoshi\\Data\\test_FFT\\Convolution_1D.txt";
	//ofilename_i << "Convolution.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "u numeric" << "\t" << "u analytic" << "\t"  << "\n";

	for (int i = 0; i < N; i++) {
		ofs_res_i << x[i] << "\t" << Convolution[i] << "\t" << Convolution_analytic[i] << "\n";
	}

	cudaFree(ft_d);
	cudaFree(gt_d);
	cudaFree(f_dc);
	cudaFree(g_dc);
	cudaFree(cvt_d);
	cudaFree(u_dc);
	cufftDestroy(plan);
	//If you delete the inexist variable, you may get error MSB 3721.
	delete[](x);
	delete[](g);
	delete[](err);
	delete[]k;
	delete[]f_Comp;
	delete[]g_Comp;
	delete[]Convolution;
	delete[]Convolution_analytic;
}

