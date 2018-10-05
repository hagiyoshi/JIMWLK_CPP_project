
#include "mkl_dfti.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/math/special_functions/bessel.hpp>
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
#include <complex>
#include <functional>
#include <random>


#include "Header_Params.h"
#include "Gellman_matrix.h"


//parameters for solving the JIMWLK equation
//we need extern for linking another object files. 
//Because const variables mean only internal likerage and don't mean to be used in the external objects
extern const double Rp = 1.0;
extern const double R_CQ = Rp / 3.0;
extern const double g2_mu_Rp = 30.0;
extern const double m_Rp = 2.0;
extern const double mass = m_Rp / Rp;
//1.0 makes the LATTICE_SIZE and NX cast double
extern const double lattice_spacing = 1.0*LATTICE_SIZE / NX;

//rapidity
double rapidity = 0;

//The matrix to evolve
//std::complex<double> matrix_V[NX*NX][3][3];


/**
* Return the matrix exponential.
*/
inline Eigen::Matrix3cd exp_U(const Eigen::Matrix3cd m) {
	// Scaling and squaring + Taylor expansion.
	const double eps = std::numeric_limits<double>::epsilon() * 5;
	const int k = 1;
	// Find s such that |m/2^s| = |m|/2^s < 1/2^k.
	double norm = std::sqrt(m.squaredNorm());
	int s = std::max((int)(std::log(norm) / std::log(2.0)) + (k + 1), 0);
	// Scaling.
	const double scale = std::pow((double)2, -s);
	Eigen::Matrix3cd a;
	a = m;
	a = a.array()*scale;
	// Taylor expansion to get exp(m/2^s).
	Eigen::Matrix3cd sum = Eigen::MatrixXd::Identity(3, 3);
	Eigen::Matrix3cd x = a;
	sum += x;
	for (int i = 2;; i++) {
		Eigen::Matrix3cd old = sum;
		x = x*a;
		double factor = 1.0 / i;
		x = x.array() * factor;
		sum += x;
		Eigen::Matrix3cd osubs = old - sum;
		double osubs_norm2 = osubs.squaredNorm();
		if (osubs_norm2 < eps * eps) {
			break;
		}
	}
	// Squaring to get exp(m) = [exp(m/2^s)]^(2^s).
	for (int i = 0; i < s; i++) {
		sum *= sum;
	}
	return sum;
}


double modified_bessel1(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI / x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return 1.0 / x;
	}
	else {
		return boost::math::cyl_bessel_k(1, x);
	}
}


double modified_bessel0(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI / x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return log(x / 2.0);
	}
	else {
		return boost::math::cyl_bessel_k(0, x);
	}
}
double modified_bessel_times_x(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI*x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return 1.0;
	}
	else {
		return x*boost::math::cyl_bessel_k(1, x);
	}
}

double Integration_kernelx(const double x, const double y)
{
	if (abs(sqrt(x*x + y * y)) < 1.0e-30) {
		return modified_bessel_times_x(mass*sqrt(x*x + y * y))*x;
	}
	else {
		return modified_bessel_times_x(mass*sqrt(x*x + y * y))*x / (x*x + y * y);
	}
}


double Integration_kernely(const double x, const double y)
{
	if (abs(sqrt(x*x + y * y)) < 1.0e-30) {
		return modified_bessel_times_x(mass*sqrt(x*x + y * y))*y;
	}
	else {
		return modified_bessel_times_x(mass*sqrt(x*x + y * y))*y / (x*x + y * y);
	}
}

void noise_generation(double* noise)
{
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());

//#pragma omp parallel for num_threads(6)
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{

			std::normal_distribution<double> dist(0.0, 1.0);
			noise[NX*j + i] = dist(engine) / (1.0*LATTICE_SIZE / NX);
		}
	}
}


double Sqrt_Constituent_Quark_Density(const double x, const double y, const double *x_CQ, const double *y_CQ)
{
	double coefficient = 1.0 / (2.0*M_PI*R_CQ*R_CQ)/3.0;
	double density = 0;
	double coeff_exp = -1.0 / (2.0*R_CQ*R_CQ);
	for (int i = 0; i < 3; i++) {
		//double x_sub_x_CQ = x - x_CQ[i];
		//double y_sub_y_CQ = y - y_CQ[i];
		density += exp(
			coeff_exp
			* ((x - x_CQ[i])*(x - x_CQ[i]) + (y - y_CQ[i])*(y - y_CQ[i]))
		);
	}

	return sqrt(coefficient*density);

}

double Sqrt_Round_Proton_Density(const double x, const double y)
{
	double coefficient = 1.0 / (2.0*M_PI*Rp*Rp);
	double density = 0;
	double coeff_exp = -1.0 / (2.0*Rp*Rp);
		//double x_sub_x_CQ = x - x_CQ[i];
		//double y_sub_y_CQ = y - y_CQ[i];
		density = exp(
			coeff_exp
			* ((x)*(x) + (y)*(y))
		);

	return sqrt(coefficient*density);

}

double Charge_density_times_g(const double x, const double y, const int GM_index, 
	const double noise, const double *x_CQ, const double *y_CQ)
{
#ifdef Round_Proton

	double coefficient = g2_mu_Rp / sqrt(INITIAL_N);

	return coefficient*Sqrt_Round_Proton_Density(x, y)*noise;
#else

	double coefficient = g2_mu_Rp/3.0 / sqrt(INITIAL_N);

	return coefficient*Sqrt_Constituent_Quark_Density(x, y, x_CQ, y_CQ)*noise;
#endif // Round_Proton

}

void Initial_quark_position(double* x_CQ,double* y_CQ)
{
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());


	for (int i_q = 0; i_q < Nc; ++i_q)
	{

		std::normal_distribution<double> dist(0.0, Rp / sqrt(2.0));
		x_CQ[i_q] = dist(engine);
		y_CQ[i_q] = dist(engine);

		//In order not to protrude the defined region
		while (abs(x_CQ[i_q]) > LATTICE_SIZE / 2.0 && abs(y_CQ[i_q]) > LATTICE_SIZE / 2.0)
		{
			std::normal_distribution<double> dist(0.0, Rp / sqrt(2.0));
			x_CQ[i_q] = dist(engine);
			y_CQ[i_q] = dist(engine);
		}
	}

}

//from JIMWLK.cu
void Solve_Poisson_Equation(double* rho,double* Solution);

//using the unsupported in Eigen. we only add the unsupported file at 
// C:\Users\hagip\source\repos\JIMWLK_CPP_project\packages\Eigen.3.3.3\build\native\include
Eigen::Matrix3cd Exp_Matrix3cd(Eigen::Matrix3cd A)
{
	Eigen::Matrix3cd Exp_A;
	Exp_A = A.exp();
	return Exp_A;
}

//Solve the Poisson equation
void Solution_times_g(double* noise_rho, double* solution_Poisson, double* x_CQ, double* y_CQ)
{

	double   xmax = lattice_spacing *NX / 2.0, xmin = -lattice_spacing*NX / 2.0, ymin = -lattice_spacing*NX / 2.0;
	double   *x = new double[NX*NX], *y = new double[NX*NX];

	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());

	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
#ifdef Round_Proton
			double coefficient = g2_mu_Rp / sqrt(INITIAL_N);

#else
			double coefficient = g2_mu_Rp/3.0 / sqrt(INITIAL_N);
#endif

			x[NX*j + i] = xmin + i*lattice_spacing;
			y[NX*j + i] = ymin + j*lattice_spacing;

			//std::normal_distribution<double> dist(0.0, 1.0 / sqrt(2.0));
			std::normal_distribution<double> dist(0.0, 1.0 );
			//noise = g2_mu_Rp / sqrt(INITIAL_N)*Sqrt_Constituent_Quark_Density(x[NX*j + i], y[NX*j + i], x_CQ, y_CQ)
			//        /(LATTICE_SIZE/NX)*gaussian_distribution
#ifdef Round_Proton
			noise_rho[NX*j + i] = coefficient*Sqrt_Round_Proton_Density(x[NX*j + i], y[NX*j + i])
				/ (1.0*LATTICE_SIZE / NX)*dist(engine);

#else
			noise_rho[NX*j + i] = coefficient*Sqrt_Constituent_Quark_Density(x[NX*j + i], y[NX*j + i], x_CQ, y_CQ) 
									/ (1.0*LATTICE_SIZE / NX)*dist(engine) ;
#endif
		}
	}

	Solve_Poisson_Equation(noise_rho, solution_Poisson); 

	delete[]x;
	delete[]y;
}

void test_fourier_noise()
{
	double impact_parameter = 0.0;
	double D_matrix=0;
	double D_matrix_MC = 0;


	double* noise_rho = new double[NX*NX];
	double* solution_Poisson = new double[NX*NX];
	double   *x = new double[NX*NX], *y = new double[NX*NX];
	double   xmax = lattice_spacing *NX / 2.0, xmin = -lattice_spacing*NX / 2.0, ymin = -lattice_spacing*NX / 2.0;
	double h = lattice_spacing;
	double   x_CQ[Nc], y_CQ[Nc];
	double relative_distance = 0.0;

	int max_num = 100;
	int positionb = NX*NX / 2 + NX / 2;

	for (int num = 0; num < max_num; num++) {
		//Initial_quark_position(x_CQ, y_CQ);

		Solution_times_g(noise_rho, solution_Poisson, x_CQ, y_CQ);
		D_matrix_MC += solution_Poisson[positionb]* solution_Poisson[positionb];
	}


	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{

			x[NX*j + i] = xmin + i*lattice_spacing;
			y[NX*j + i] = ymin + j*lattice_spacing;

			double simpson1 = 1.0;
			double simpson2 = 1.0;
			if (j == 0 || j == NX - 1) {
				simpson1 = 1.0 / 3.0;
			}
			else if (j % 2 == 0) {
				simpson1 = 2.0 / 3.0;
			}
			else {
				simpson1 = 4.0 / 3.0;
			}

			if (i == 0 || i == NX - 1) {
				simpson2 = 1.0 / 3.0;
			}
			else if (i % 2 == 0) {
				simpson2 = 2.0 / 3.0;
			}
			else {
				simpson2 = 4.0 / 3.0;
			}
			if (abs((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
				+ (y[NX*j + i] + relative_distance)* (y[NX*j + i] + relative_distance)) < 1.0e-16
				) {

				D_matrix += 0;
			}
			else {
				D_matrix += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 2.0 / Rp / Rp)
					*(modified_bessel0(mass
						*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
							+ (y[NX*j + i] + relative_distance)* (y[NX*j + i] + relative_distance)))
						)
					*(modified_bessel0(mass
						*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
							+ (y[NX*j + i] + relative_distance)* (y[NX*j + i] + relative_distance)))
						);


			}

		}

	}

	double exp_coeff = g2_mu_Rp*g2_mu_Rp/ (2.0*M_PI) / (2.0*M_PI) / (2.0*M_PI);

	std::cout << "MC \t" << D_matrix_MC / (double(max_num))*(double(INITIAL_N)) << " \t analytical \t" << exp_coeff*D_matrix*h*h << "\n";

		delete[]noise_rho;
		delete[]solution_Poisson;
		delete[]x;
		delete[]y;
}

//A matrix represented as a multidimentional array is passed as a pointer
//Calculating the initial condition for the JIMWLK equation
void Calculate_initial_condition_wo_stack_overflow(std::complex<double>* V_init, double* x_CQ, double* y_CQ)
{

	double* noise_rho = new double[NX*NX];
	double* solution_Poisson = new double[NX*NX];


	//std::complex<double> V_initial[NX*NX][3][3];

	std::complex<double> minus_pure_imaginary(0.0, -1.0);

	for (int in = 0; in < INITIAL_N; ++in) {

		std::complex<double>* V_gauss = new std::complex<double>[3*3*NX*NX];

		for (int a = 0; a < ADJNc; ++a) {

			Solution_times_g(noise_rho, solution_Poisson, x_CQ, y_CQ);

			for (int vx = 0; vx < NX*NX; ++vx) {


				std::complex<double> solution_comp(solution_Poisson[vx], 0.0);

				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						V_gauss[3*3*vx+3*i+j] += minus_pure_imaginary*Generator_adjt[a][i][j] * solution_comp;
					}
				}
			}

		}

#pragma omp parallel for num_threads(6)
		for (int vx = 0; vx < NX*NX; ++vx) {

			Eigen::Matrix3cd A, A_exp;

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					A(i, j) = V_gauss[3 * 3 * vx + 3 * i + j];
				}
			}

			if (in == 0) {
				//A_exp = A.exp();
				A_exp = exp_U(A);

				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						//A(i, j) = V_gauss[3 * 3 * vx + 3 * i + j];
						V_init[3 * 3 * vx + 3 * i + j] = A_exp(i, j);
					}
				}
			}
			else {

				//A_exp = A.exp();
				A_exp = exp_U(A);
				Eigen::Matrix3cd B,D;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						B(i, j) = V_init[3 * 3 * vx + 3 * i + j];
					}
				}

				D = B*A_exp;

				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						V_init[3 * 3 * vx + 3 * i + j] = D(i, j);
					}
				}


			}


		}

		delete[]V_gauss;
		std::cout << "initial n " << in << "\n";
	}


	delete[]noise_rho;
	delete[]solution_Poisson;
}

//Calculation of 2 dimentional convolution
void Calculate_Convolution(double* func1, double* func2, double* Convolution);


void convolve_f_and_g(std::complex<double>* ft, std::complex<double>* gt, std::complex<double>* cvt_k, int N)
{
#pragma omp parallel for num_threads(6)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = j * N + i;
				if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
					cvt_k[index] = std::complex<double>((ft[index].real() * gt[index].real() *1.0 - ft[index].imag() * gt[index].imag() *1.0),
						(ft[index].imag() * gt[index].real() *1.0 + ft[index].real() * gt[index].imag() *1.0));
				}
				else {

					cvt_k[index] = std::complex<double>(-(ft[index].real() * gt[index].real() *1.0 - ft[index].imag() * gt[index].imag() *1.0),
						-(ft[index].imag() * gt[index].real() *1.0 + ft[index].real() * gt[index].imag() *1.0));
				}
		}
	}
}


void Calculate_Convolution_MKL(double* func1, double* func2, double* Convolution) {
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


	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	MKL_LONG status, l[2];
	l[0] = NX; l[1] = NX;
	status = DftiCreateDescriptor(&my_desc1_handle, DFTI_SINGLE,
		DFTI_COMPLEX, 2, l);

	status = DftiCommitDescriptor(my_desc1_handle);
	status = DftiComputeForward(my_desc1_handle, f_Comp);
	status = DftiComputeForward(my_desc1_handle, g_Comp);

	std::complex<double>* Conv_Comp = new std::complex<double>[N*N];
	convolve_f_and_g(f_Comp, g_Comp, Conv_Comp, N);

	status = DftiComputeBackward(my_desc1_handle, Conv_Comp);

	status = DftiFreeDescriptor(&my_desc1_handle);

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			Convolution[N*j + i] = Conv_Comp[N*j + i].real() * 1.0*LATTICE_SIZE*1.0*LATTICE_SIZE / ((double)(N*N)) / ((double)(N*N));
		}
	}

	delete[](x);
	delete[](y);
	delete[](f);
	delete[](g);
	delete[](u_a);
	delete[](err);
	delete[]f_Comp;
	delete[]g_Comp;
	delete[]Conv_Comp;
}


void Calculate_Convolution_complex(std::complex<double>* func1, std::complex<double>* func2);

void Calculation_2D_convolution()
{


	int N = NX;
	double* x = new double[NX*NX];
	double* y = new double[NX*NX];
	double* f = new double[NX*NX];
	double* g = new double[NX*NX];
	double* Convolution = new double[NX*NX];
	double* analytical_solution = new double[NX*NX];
	double h = lattice_spacing;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			f[N*j + i] = exp(-(x[N*j + i] * x[N*j + i] + y[N*j + i] * y[N*j + i]) / 3.0);
			g[N*j + i] = exp(-(x[N*j + i] * x[N*j + i] + y[N*j + i] * y[N*j + i]) / 2.0);
			analytical_solution[N*j + i] = (6.0 / 5.0)*exp(-(x[N*j + i] * x[N*j + i] + y[N*j + i] * y[N*j + i]) / 5.0)*M_PI;
		}
	}

	//Calculate_Convolution(f, g, Convolution);
	Calculate_Convolution_MKL(f, g, Convolution);

	std::ostringstream ofilename_c;
	ofilename_c << "G:\\hagiyoshi\\Data\\test_FFT\\Convolution_test.txt";
	std::ofstream ofs_res_c(ofilename_c.str().c_str());

	ofs_res_c << "#x" << "\t" << "y" << "\t" << "u numeric" << "\t" << "u analytic" 
		<<"\t" << "u analytic / u numeric" << "\n";


	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			ofs_res_c << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Convolution[N*j + i] << "\t" << analytical_solution[N*j + i]
				<< "\t" << analytical_solution[N*j + i]/ Convolution[N*j + i] << "\n";
		}
		ofs_res_c << "\n";
	}

	delete[]x;
	delete[]y;
	delete[]f;
	delete[]g;
	delete[]Convolution;
	delete[]analytical_solution;

}

void Calculate_Convolution_1D();


void One_step_matrix(std::complex<double>* V_initial,double delta_rapidity)
{

	std::vector<double> x(NX*NX, 0);
	std::vector<double> y(NX*NX, 0);
	double   xmax = lattice_spacing *NX / 2.0, xmin = -lattice_spacing*NX / 2.0, ymin = -lattice_spacing*NX / 2.0;
	std::vector<std::complex<double>>  kernel1(NX*NX, 0);
	std::vector<std::complex<double>>  kernel2(NX*NX, 0);
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*lattice_spacing;
			y[NX*j + i] = ymin + j*lattice_spacing;
			//define integration kernel
			std::complex<double> kernelx(Integration_kernelx(x[NX*j + i], y[NX*j + i]), 0.0);
			std::complex<double> kernely(Integration_kernely(x[NX*j + i], y[NX*j + i]), 0.0);
			kernel1[NX*j + i] = kernelx;
			kernel2[NX*j + i] = kernely;
		}
	}

	//  xi matrix
	std::vector<std::complex<double>> V_gauss1(3 * 3 * NX*NX, 0);
	std::vector<std::complex<double>> V_gauss2(3 * 3 * NX*NX, 0);
	std::vector<double> noise_1(NX*NX, 0);
	std::vector<double> noise_2(NX*NX, 0);

	for (int a = 0; a < ADJNc; ++a) {

		noise_generation(noise_1.data());
		noise_generation(noise_2.data());

		for (int vx = 0; vx < NX*NX; ++vx) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					//define xi_z
					V_gauss1[3 * 3 * vx + 3 * i + j] += Generator_adjt[a][i][j] * noise_1[vx];
					V_gauss2[3 * 3 * vx + 3 * i + j] += Generator_adjt[a][i][j] * noise_2[vx];

				}
			}

		}

	}

	//V*xi*V^dagger matrix
	std::vector<std::complex<double>> V_gaussf1(3 * 3 * NX*NX, 0);
	std::vector<std::complex<double>> V_gaussf2(3 * 3 * NX*NX, 0);
	Eigen::Matrix3cd A_ini,A_ini_adj, B1, C2, VxiVd1, VxiVd2,test_unitary;
	for (int vx = 0; vx < NX*NX; ++vx) {

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				//define V, xi_1, xi_2
				A_ini(i, j) = V_initial[3 * 3 * vx + 3 * i + j];
				B1(i, j) = V_gauss1[3 * 3 * vx + 3 * i + j];
				C2(i, j) = V_gauss2[3 * 3 * vx + 3 * i + j];

			}
		}

		A_ini_adj = A_ini.adjoint();

		// V xi V^dagger
		VxiVd1 = A_ini*B1*A_ini_adj;
		VxiVd2 = A_ini*C2*A_ini_adj;
		test_unitary = A_ini*A_ini_adj;

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				V_gaussf1[3 * 3 * vx + 3 * i + j] = VxiVd1(i, j);
				V_gaussf2[3 * 3 * vx + 3 * i + j] = VxiVd2(i, j);

			}
		}

	}

	//integrate kernel * xi
	std::vector<std::complex<double>> V_convf1(NX*NX, 0);
	std::vector<std::complex<double>> V_convf2(NX*NX, 0);
	std::vector<std::complex<double>> V_conv1(NX*NX, 0);
	std::vector<std::complex<double>> V_conv2(NX*NX, 0);

	//std::vector<std::complex<double>> V_after_convf1(NX*NX, 0);
	//std::vector<std::complex<double>> V_after_convf2(NX*NX, 0);
	//std::vector<std::complex<double>> V_after_conv1(NX*NX, 0);
	//std::vector<std::complex<double>> V_after_conv2(NX*NX, 0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {

			for (int vx = 0; vx < NX*NX; ++vx) {
				V_conv1[vx] = V_gauss1[3 * 3 * vx + 3 * i + j];
				V_conv2[vx] = V_gauss2[3 * 3 * vx + 3 * i + j];
				V_convf1[vx] = V_gaussf1[3 * 3 * vx + 3 * i + j];
				V_convf2[vx] = V_gaussf2[3 * 3 * vx + 3 * i + j];
			}

			Calculate_Convolution_complex(kernel1.data(), V_conv1.data());
			Calculate_Convolution_complex(kernel2.data(), V_conv2.data());
			Calculate_Convolution_complex(kernel1.data(), V_convf1.data());
			Calculate_Convolution_complex(kernel2.data(), V_convf2.data());


			for (int vx = 0; vx < NX*NX; ++vx) {
				V_gauss1[3 * 3 * vx + 3 * i + j] = V_conv1[vx];
				V_gauss2[3 * 3 * vx + 3 * i + j] = V_conv2[vx];
				V_gaussf1[3 * 3 * vx + 3 * i + j] = V_convf1[vx];
				V_gaussf2[3 * 3 * vx + 3 * i + j] = V_convf2[vx];
			}

		}
	}


	std::complex<double> coeff(0.0, -1.0*sqrt(ALPHA_S*delta_rapidity) / M_PI);
	std::complex<double> coeff2(0.0, 1.0*sqrt(ALPHA_S*delta_rapidity) / M_PI);

#pragma omp parallel for num_threads(6)
	for (int vx = 0; vx < NX*NX; ++vx) {

		//you should place the definition of the matrixes in the forloop so as to define them in each thread.
		Eigen::Matrix3cd A_f, B_b, A_exp, B_exp, C;
		Eigen::Matrix3cd B, D;

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				A_f(i, j) = coeff*(V_gaussf1[3 * 3 * vx + 3 * i + j] + V_gaussf2[3 * 3 * vx + 3 * i + j]);
				B_b(i, j) = coeff2*(V_gauss1[3 * 3 * vx + 3 * i + j] + V_gauss2[3 * 3 * vx + 3 * i + j]);
			}
		}

		//A_exp = A_f.exp();
		//B_exp = B_b.exp();
		A_exp = exp_U(A_f);
		B_exp = exp_U(B_b);

		C = A_exp*B_exp;

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				B(i, j) = V_initial[3 * 3 * vx + 3 * i + j];
				//V_initial[3 * 3 * vx + 3 * i + j] = (A_exp*B*B_exp)(i, j);
			}
		}

		//if we directly V_initial = (A_exp*B*B_exp)(i,j) , we have wrong results.
		D = A_exp*B*B_exp;

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				V_initial[3 * 3 * vx + 3 * i + j] = D(i, j);
			}
		}

	}

	rapidity += delta_rapidity;
}

//print Re(1-V(x))/N_c
void print_1_V(std::complex<double>* V_init, double* x_CQ, double* y_CQ, const double number_rapidity,const int number)
{

	double* trace_1_V = new double[NX*NX];
	for (int n = 0; n < NX*NX; ++n) {
		trace_1_V[n] = 0;
		for (int i = 0; i < 3; ++i) {
			trace_1_V[n] += (1.0 - 1.0*V_init[3 * 3 * n + 3 * i + i].real()) / Nc;
		}

	}


	std::ostringstream ofilename_i, ofilename_V;
#ifdef v_Parameter
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\re1_v_over_wvpara_Nc_num_" << number << "_NX_" << NX << "_INITN_" << INITIAL_N 
						<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
	ofilename_V << "G:\\hagiyoshi\\Data\\JIMWLK\\test_vParam\\matrix_V_wvpara_num_" << number << "_" << NX << "_INITN_" << INITIAL_N 
						<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
#else
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\re1_v_over_Nc_num_" << number << "_NX_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
	ofilename_V << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_num_"<< number << "_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
#endif
	//ofilename_i << "solution_Poisson_cpp.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());
	std::ofstream ofs_res_V(ofilename_V.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t" << "\n";

	ofs_res_V << "#x_CQ" << "\t" << "y_CQ" << "\n";
	for (int i = 0; i < 3; i++) {
		//double x_sub_x_CQ = x - x_CQ[i];
		//double y_sub_y_CQ = y - y_CQ[i];
		ofs_res_V<<"# " << x_CQ[i] << "\t" << y_CQ[i] << "\n";
	}

	//std::cout << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t"  << "\n";
	int N = NX;
	double* x = new double[NX*NX];
	double* y = new double[NX*NX];
	double h = lattice_spacing;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << trace_1_V[N*j + i] << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}

	for (int vx = 0; vx < NX*NX; ++vx) {

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				//define V, xi_1, xi_2
				ofs_res_V << V_init[3 * 3 * vx + 3 * i + j] <<"\n";

			}
		}
	}


	if (number_rapidity < EPS) {
		std::ostringstream ofilename_i2;
#ifdef v_Parameter
		ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\test_vParam\\CQ_position_num_"<< number << "_NX_" << NX << "_INITN_" << INITIAL_N 
							<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
#else
		ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\CQ_position_num_" << number << "_NX_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
#endif
		//ofilename_i << "solution_Poisson_cpp.txt";
		std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

		ofs_res_i2 << "#x_CQ" << "\t" << "y_CQ" << "\n";
		for (int i = 0; i < 3; i++) {
			//double x_sub_x_CQ = x - x_CQ[i];
			//double y_sub_y_CQ = y - y_CQ[i];
			ofs_res_i2 << x_CQ[i] << "\t" << y_CQ[i] << "\n";
		}
	}

	delete[]x;
	delete[]y;
	delete[]trace_1_V;
}


//print Re(1-V(x))/N_c
void print_1_V_round_proton(std::complex<double>* V_init, const double number_rapidity, const int number)
{

	double* trace_1_V = new double[NX*NX];
	for (int n = 0; n < NX*NX; ++n) {
		trace_1_V[n] = 0;
		for (int i = 0; i < 3; ++i) {
			trace_1_V[n] += (1.0 - 1.0*V_init[3 * 3 * n + 3 * i + i].real()) / Nc;
		}

	}


	std::ostringstream ofilename_i, ofilename_V;
#ifdef v_Parameter
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\re1_v_over_wvpara_Nc_RP_num_" << number << "_NX_" << NX << "_INITN_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
	ofilename_V << "G:\\hagiyoshi\\Data\\JIMWLK\\test_vParam\\matrix_V_wvpara_RP_num_" << number << "_" << NX << "_INITN_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
#else
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\re1_v_over_Nc_RP_num_" << number << "_NX_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
	ofilename_V << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_RP_num_" << number << "_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
#endif
	//ofilename_i << "solution_Poisson_cpp.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());
	std::ofstream ofs_res_V(ofilename_V.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t" << "\n";


	//std::cout << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t"  << "\n";
	int N = NX;
	double* x = new double[NX*NX];
	double* y = new double[NX*NX];
	double h = lattice_spacing;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << trace_1_V[N*j + i] << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}

	for (int vx = 0; vx < NX*NX; ++vx) {

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				//define V, xi_1, xi_2
				ofs_res_V << V_init[3 * 3 * vx + 3 * i + j] << "\n";

			}
		}
	}


	

	delete[]x;
	delete[]y;
	delete[]trace_1_V;
}



void print_1_V_initial_unit(std::complex<double>* V_init, const double number_rapidity)
{

	double* trace_1_V = new double[NX*NX];
	for (int n = 0; n < NX*NX; ++n) {
		trace_1_V[n] = 0;
		for (int i = 0; i < 3; ++i) {
			trace_1_V[n] += (1.0 - 1.0*V_init[3 * 3 * n + 3 * i + i].real()) / Nc;
		}

	}


	std::ostringstream ofilename_i;
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\re1_v_over_Nc_initial_unit_NX_" 
		<< NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
	//ofilename_i << "solution_Poisson_cpp.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t" << "\n";

	//std::cout << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t"  << "\n";
	int N = NX;
	double* x = new double[NX*NX];
	double* y = new double[NX*NX];
	double h = lattice_spacing;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0;
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i*h;
			y[N*j + i] = ymin + j*h;
			ofs_res_i << x[N*j + i] << "\t" << y[N*j + i] << "\t" << trace_1_V[N*j + i] << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		ofs_res_i << "\n";
		//std::cout << "\n";
	}


	delete[]x;
	delete[]y;
	delete[]trace_1_V;
}


void print_1_V_initial_unit_outside_sqare_proton(std::complex<double>* V_init, const double number_rapidity, const int number)
{

	std::complex<double>* trace_V = new std::complex<double>[NX*NX];
	for (int n = 0; n < NX*NX; ++n) {
		trace_V[n] = 0;
		for (int i = 0; i < 3; ++i) {
			trace_V[n] += (1.0*V_init[3 * 3 * n + 3 * i + i]);
		}

	}


	std::ostringstream ofilename_i;
	ofilename_i << "G:\\hagiyoshi\\Data\\JIMWLK\\test_square\\re_v_over_Nc_expu_initial_unit_outside_sqare_proton_NX_"
		<< NX << "_INITN_" << INITIAL_N << "_num_" << number << "_" << number_rapidity<< ".txt";
	//ofilename_i << "solution_Poisson_cpp.txt";
	std::ofstream ofs_res_i(ofilename_i.str().c_str());

	ofs_res_i << "#x" << "\t" << "y" << "\t" << "Re(V)/Nc" << "\t" << "\n";

	//std::cout << "#x" << "\t" << "y" << "\t" << "Re(1-V)/Nc" << "\t"  << "\n";
	int N = NX;
	double* x = new double[NX*NX];
	double* y = new double[NX*NX];
	double h = lattice_spacing;
	double   xmax = h * N / 2.0, xmin = -h * N / 2.0, ymin = -h * N / 2.0;
	std::complex<double> numo3(1.0/3.0, 1.0 / 3.0);
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++)
		{
			x[N*j + i] = xmin + i * h;
			y[N*j + i] = ymin + j * h;
			ofs_res_i 
				//<< x[N*j + i] << "\t" << y[N*j + i] << "\t" 
				<< trace_V[N*j + i]*numo3 << "\n";
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		//ofs_res_i << "\n";
		//std::cout << "\n";
	}


	delete[]x;
	delete[]y;
	delete[]trace_V;
}

//generate 3*3 unit matrix
void Initialize_unit_matrix(std::complex<double>* V_initial)
{
	Eigen::Matrix3cd A;
	A = Eigen::MatrixXd::Identity(3,3);

	for (int vx = 0; vx < NX*NX; ++vx) {

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				V_initial[3 * 3 * vx + 3 * i + j] = A(i,j);

			}
		}

	}

}

//generate unitary matrixes inside the square and unit matrixes at the outside.
void Initialize_unit_matrix_outside_sqare_proton(std::complex<double>* V_initial)
{
	Eigen::Matrix3cd A, Unitary_example;
	A = Eigen::MatrixXd::Identity(3, 3);
	//tr(Unitary_example) = -1, Unitary_example*Unitary_example^dagger = Identity
	Unitary_example << std::complex<double>(-1.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
		std::complex<double>(0.0, 0.0), std::complex<double>(-1.0, 0.0), std::complex<double>(0.0, 0.0),
		std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0);


	for (int ix = 0; ix < NX; ++ix) {
		for (int jy = 0; jy < NX; ++jy) {

			if (ix > NX / 3.0 && ix<2.0*NX / 3.0 && jy>NX / 3.0 && jy < 2.0*NX / 3.0) {

				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						V_initial[3 * 3 * (NX*ix + jy) + 3 * i + j] = Unitary_example(i, j);

					}
				}

			}
			else {


				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						V_initial[3 * 3 * (NX*ix + jy) + 3 * i + j] = A(i, j);

					}
				}
			}

		}

	}

}


int main()
{


	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	//test_fourier_noise();
	//Calculate_Convolution_1D();
	Calculation_2D_convolution();
	Generator_SU3_initializer();
	double   x_CQ[Nc], y_CQ[Nc];

	for (int num = 161; num <= 190; num++) {

		rapidity = 0.0;

		//locate the quark in the proton
		Initial_quark_position(x_CQ, y_CQ);

		Calculate_initial_condition_wo_stack_overflow(V_initial, x_CQ, y_CQ);
		//Initialize_unit_matrix( V_initial);
		//Initialize_unit_matrix_outside_sqare_proton(V_initial);

		double next_rapidity = 0.0;

#ifdef Round_Proton

		print_1_V_round_proton(V_initial, rapidity, num);
#else

		print_1_V(V_initial, x_CQ, y_CQ, rapidity, num);
#endif // Round_Proton
		//print_1_V_initial_unit(V_initial,rapidity);
		//print_1_V_initial_unit_outside_sqare_proton(V_initial, rapidity, num);

#ifdef EVOLUTION
		for (;;) {
			int reunit_count = 0;
			if (rapidity >= END_Y*1.0 - EPS) {
				break;
			}
			next_rapidity = std::min(next_rapidity + OUTPUT_Y, 1.0*END_Y);
			while (rapidity < next_rapidity - EPS) {
				One_step_matrix(V_initial, std::min(DELTA_Y, next_rapidity - rapidity));
			}
#ifdef Round_Proton

			print_1_V_round_proton(V_initial, rapidity, num);
#else

			print_1_V(V_initial, x_CQ, y_CQ, rapidity, num);
#endif // Round_Proton


			//print_1_V_initial_unit(V_initial, rapidity);
			//print_1_V_initial_unit_outside_sqare_proton(V_initial, rapidity, num);
		}


#ifdef Round_Proton

		print_1_V_round_proton(V_initial, rapidity, num);
#else

		print_1_V(V_initial, x_CQ, y_CQ, rapidity, num);
#endif // Round_Proton

		//print_1_V_initial_unit(V_initial, rapidity);
		//print_1_V_initial_unit_outside_sqare_proton(V_initial, rapidity, num);
		//Calculation_2D_convolution();
#endif
	}
	delete[]V_initial;
}