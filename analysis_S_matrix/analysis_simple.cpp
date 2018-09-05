#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/math/special_functions/bessel.hpp>
#include <thrust/random/normal_distribution.h>
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
#include "interpolate_2d_array.h"
#include "Spline.h"


extern const double Rp = 1.0;
extern const double R_CQ = Rp / 3.0;
extern const double g2_mu_Rp = 30.0;
extern const double m_Rp = 2.0;
extern const double mass = m_Rp / Rp;

int number_of_comfig = 500;
int initial_number = 0;
//lattice rotational symmetry -||-
int number_of_symmetry = 4;

const double Lamda_QCD_nucleus = (Lamda_QCD / 1.0);
//const double Lamda_QCD_nucleus = 1.0;


double modified_bessel1(const double x)
{
	double x_times_k_1 = 0;
	if (abs(x) > 900.0) {
		return sqrt(M_PI/x / 2.0)*exp(-x);
	}
	else if (abs(x) < 1.0e-30)
	{
		return 1.0/x;
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
		return log(x/2.0);
	}
	else {
		return boost::math::cyl_bessel_k(0, x);
	}
}

void MV_model_calculation_of_T_matrix()
{
	double impact_parameter = 1.0*NX / LATTICE_SIZE;
	impact_parameter = 128;

	std::vector<double> D_matrix(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0*re*h;
	}

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

#pragma omp parallel for num_threads(6)
	for (int re = 1; re < NX / 2; re++) {

		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++) {
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
					+ (y[NX*j + i] - relative_distance[re] / 2.0)* (y[NX*j + i] - relative_distance[re] / 2.0)) < 1.0e-16
					|| abs((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
						+ (y[NX*j + i] + relative_distance[re] / 2.0)* (y[NX*j + i] + relative_distance[re] / 2.0)) < 1.0e-16) {

					D_matrix[re] += 0;
				}
				else {
					D_matrix[re] += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 2.0 / Rp / Rp)
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
								+ (y[NX*j + i] - relative_distance[re] / 2.0)* (y[NX*j + i] - relative_distance[re] / 2.0)))
							- modified_bessel0(mass
								*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
									+ (y[NX*j + i] + relative_distance[re] / 2.0)* (y[NX*j + i] + relative_distance[re] / 2.0))))
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
								+ (y[NX*j + i] - relative_distance[re] / 2.0)* (y[NX*j + i] - relative_distance[re] / 2.0)))
							- modified_bessel0(mass
								*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
									+ (y[NX*j + i] + relative_distance[re] / 2.0)* (y[NX*j + i] + relative_distance[re] / 2.0))));
										
				}
			}
		}
	}

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter <<  ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX<< ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter <<  ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_MV_position_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t D_matrix \t position " << impact_parameter << "\n";

	double exp_coeff = -g2_mu_Rp*g2_mu_Rp*(Nc*Nc - 1.0) / 4.0 / Nc / (2.0*M_PI) / (2.0*M_PI) / (2.0*M_PI);

	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << 1.0 - exp(exp_coeff*h*h*D_matrix[re]) << "\t" <<D_matrix[re] << "\n";
	}


}


void MV_model_calculation_of_V_matrix(int position_B)
{
	double impact_parameter = 1.0*NX / LATTICE_SIZE;
	impact_parameter = position_B;

	std::vector<double> D_matrix(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = re*h;
	}

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

#pragma omp parallel for num_threads(6)
	for (int re = 0; re < NX / 2; re++) {

		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++) {
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
					+ (y[NX*j + i] + relative_distance[re] )* (y[NX*j + i] + relative_distance[re] )) < 1.0e-16
					) {

					D_matrix[re] += 0;
				}
				else {
					D_matrix[re] += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 2.0 / Rp / Rp)
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
								+ (y[NX*j + i] + relative_distance[re] )* (y[NX*j + i] + relative_distance[re] )))
							)
						*(modified_bessel0(mass
							*sqrt((x[NX*j + i] - impact_parameter*h)*(x[NX*j + i] - impact_parameter*h)
								+ (y[NX*j + i] + relative_distance[re] )* (y[NX*j + i] + relative_distance[re] )))
							);

				}
			}
		}
	}

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_MV_position_RP_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_MV_position_" << impact_parameter << "_NX_" << NX
		<< "_vPara_" << v_Parameter << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_MV_position_" << impact_parameter << "_NX_" << NX << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t V_matrix \t position " << impact_parameter << "\n";

	double exp_coeff = -g2_mu_Rp*g2_mu_Rp*(Nc*Nc - 1.0) / 4.0 / Nc / (2.0*M_PI) / (2.0*M_PI) / (2.0*M_PI);

	for (int re = 0; re < NX / 2; ++re) {
		ofs_res_i2 << 2.0*relative_distance[re] << "\t" << 1.0 - exp(exp_coeff*h*h*D_matrix[re]) << "\t" << D_matrix[re] << "\n";
	}


}


void g4times_Gamma_functions(double position_x1, double position_x2, double position_y1, double position_y2, double* gamma_functions)
{
	double dd_gamma = 0;
	double d_gamma_d_gamma_x1 = 0;
	double d_gamma_d_gamma_x2 = 0;
	double d_gamma_d_gamma_y1 = 0;
	double d_gamma_d_gamma_y2 = 0;
	double Gamma = 0;


	double h = 1.0*LATTICE_SIZE / NX;


	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}


	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++) {
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
			if (abs((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
				+ (position_x2 - y[NX*j + i] )*(position_x2 - y[NX*j + i])) < 1.0e-16
				|| abs((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
					+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])) < 1.0e-16) {

				dd_gamma += 0;
				d_gamma_d_gamma_x1 += 0;
				d_gamma_d_gamma_x2 += 0;
				d_gamma_d_gamma_y1 += 0;
				d_gamma_d_gamma_y2 += 0;
				Gamma += 0;
			}
			else {
				dd_gamma += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*((position_x1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])));


				d_gamma_d_gamma_x1 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_x1 - x[NX*j + i])
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));
				d_gamma_d_gamma_x2 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_x2 - y[NX*j + i])
					/ sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
						+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));
				d_gamma_d_gamma_y1 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_y1 - x[NX*j + i])
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
								+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))));
				d_gamma_d_gamma_y2 += simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(position_y2 - y[NX*j + i])
					/ sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
						+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))
					*modified_bessel1(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
					*(-modified_bessel0(mass
						*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
							+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i])))
						+ modified_bessel0(mass
							*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
								+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i]))));

				Gamma += -simpson1*simpson2*exp(-(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) / 1.0 / Rp / Rp)
					*(modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						- modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))))
					*(modified_bessel0(mass
						*sqrt((position_x1 - x[NX*j + i])*(position_x1 - x[NX*j + i])
							+ (position_x2 - y[NX*j + i])*(position_x2 - y[NX*j + i])))
						- modified_bessel0(mass
							*sqrt((position_y1 - x[NX*j + i])*(position_y1 - x[NX*j + i])
								+ (position_y2 - y[NX*j + i])*(position_y2 - y[NX*j + i]))));

			}

		}
	}

	gamma_functions[0] = g2_mu_Rp*g2_mu_Rp*dd_gamma*mass*mass / 2.0 / M_PI / 2.0 / M_PI*h*h;
	gamma_functions[1] = g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_x1*mass / M_PI*h*h*g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_y1*mass / M_PI*h*h
		+ g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_x2*mass / M_PI*h*h*g2_mu_Rp*g2_mu_Rp*d_gamma_d_gamma_y2*mass / M_PI*h*h;
	gamma_functions[2] = g2_mu_Rp*g2_mu_Rp*Gamma / 2.0 / M_PI / 2.0 / M_PI*h*h;

}

void MV_Wigner(int maxmom)
{
	vector<double> g4gammas(3, 0);
	std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2, 0), WignerS(maxmom*NX / 2, 0), EWignerS(maxmom*NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	for (int mom = 0; mom < maxmom; mom++) {
		double momk = P_UPPER / maxmom*mom;


#pragma omp parallel for num_threads(6)
		for (int ip = 1; ip <= NX / 2; ip++) {
			double impact_parameter = ip*h;

			for (int j = 0; j < NX; j++) {
				for (int i = 0; i < NX; i++)
				{
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

					if (abs(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]) < 1.0e-16) {
						WignerS[mom*NX / 2 + ip - 1] += 0;
						EWignerS[mom*NX / 2 + ip - 1] += 0;
					}
					else {
						g4times_Gamma_functions(impact_parameter + x[NX*j + i], y[NX*j + i], 
							impact_parameter - x[NX*j + i], - y[NX*j + i], g4gammas.data());
						WignerS[mom*NX / 2 + ip - 1] += simpson1*simpson2
							*_j0(2.0*momk*sqrt(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]))
							*((Nc*Nc-1.0)/2.0/Nc * g4gammas[0]
								+ (Nc*Nc - 1.0) / 2.0 / Nc*(Nc*Nc - 1.0) / 2.0 / Nc/4.0*g4gammas[1]
								)
							*exp(-x[NX*j + i] * x[NX*j + i] - y[NX*j + i] * y[NX*j + i] )
							*exp((Nc*Nc - 1.0) / 2.0 / Nc/2.0*g4gammas[2])
							;
						EWignerS[mom*NX / 2 + ip - 1] += simpson1*simpson2
							*_j0(2.0*momk*sqrt(x[NX*j + i] * x[NX*j + i] + y[NX*j + i] * y[NX*j + i]))
							*exp(-x[NX*j + i] * x[NX*j + i] - y[NX*j + i] * y[NX*j + i])
							*g4gammas[0]/ g4gammas[2]
							*( exp(Nc/2.0*g4gammas[2]) - 1.0);
					}

				}


			}

		}

		std::cout << "mom" << mom << "\n";
	}

	std::ostringstream ofilename_Wigner, ofilename_Wigner_all;
	ofilename_Wigner << "DP_WW_Wigner0_MV_diag_direct_NX_" << NX << "_size_" << LATTICE_SIZE
		 << "_real.txt";
	std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

	ofs_res_Wigner << "#b \t momk \t DP \t WW \n";

	for (int mom = 0; mom < maxmom; mom++) {
		double momk = P_UPPER / maxmom*mom;
		for (int j = 0; j < NX / 2; j++) {

			ofs_res_Wigner << (j+1)*h << "\t" << momk << "\t" << 2.0*Nc/ALPHA_S/M_PI/M_PI* WignerS[NX / 2 * mom + j]*h*h 
				<< "\t" << 4.0*(Nc*Nc - 1) / 2.0 / Nc /ALPHA_S / M_PI / M_PI*EWignerS[NX / 2 * mom + j] * h*h << "\n";
		}
		ofs_res_Wigner << "\n";
	}


}

void Load_matrix_V(std::complex<double>* V_matrix, const double number_rapidity, const int number)
{

	std::ostringstream ifilename;
#ifdef Round_Proton

#ifdef v_Parameter
	ifilename << "G:\\hagiyoshi\\Data\\JIMWLK\\test_vParam\\matrix_V_wvpara_RP_num_" << number << "_" << NX << "_INITN_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
	//ifilename << "E:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_num_5_256_INITN_100_0.txt";
#else
	ifilename << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_RP_num_" << number << "_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
#endif

#else


#ifdef v_Parameter
	ifilename << "G:\\hagiyoshi\\Data\\JIMWLK\\test_vParam\\matrix_V_wvpara_num_" << number << "_" << NX << "_INITN_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_" << number_rapidity << ".txt";
	//ifilename << "E:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_num_5_256_INITN_100_0.txt";
#else
	ifilename << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\matrix_V_num_" << number << "_" << NX << "_INITN_" << INITIAL_N << "_" << number_rapidity << ".txt";
#endif

#endif
	//imput and output file
	std::ifstream ifs(ifilename.str().c_str());

	char str[256];
	if (ifs.fail())
	{
		std::cerr << "failed to load file "<< number << "\t" << number_rapidity << std::endl;
	}
#ifndef Round_Proton
	for (int i = 0; i < 4; ++i) {
		ifs.getline(str, 256 - 1);
	}
#endif // !Round_Proton


	for (int vx = 0; vx < NX*NX; ++vx) {

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				//define V, xi_1, xi_2
				ifs >> V_matrix[3 * 3 * vx + 3 * i + j];

			}
		}
	}
}

void assemble_initial_Quark_position( const int number_upper)
{

	std::vector<double> x_CQ, y_CQ;

	for (int i = 0; i < number_upper; ++i) {
		std::ostringstream ofilename_i2;
		ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\JIMWLK_matrix\\CQ_position_num_" << i << "_NX_" << NX << "_INITN_" << INITIAL_N << "_" << 0 << ".txt";
		std::ifstream ifs_res_i2(ofilename_i2.str().c_str());

		char str[256];
		if (ifs_res_i2.fail())
		{
			std::cerr << "failed to load file" << std::endl;
		}
		ifs_res_i2.getline(str, 256 - 1);

		for (int i = 0; i < 3; i++) {
			//double x_sub_x_CQ = x - x_CQ[i];
			//double y_sub_y_CQ = y - y_CQ[i];
			double x, y;
			ifs_res_i2 >> x  >> y;
			x_CQ.push_back(x);
			y_CQ.push_back(y);
		}

	}

	std::ostringstream output;
	output << "G:\\hagiyoshi\\Data\\JIMWLK\\CQ_position_NX_" << NX << "_INITN_" << INITIAL_N << ".txt";
	std::ofstream ofs(output.str().c_str());

	for (int i = 0; i < x_CQ.size(); ++i) {
		ofs << x_CQ[i] << "\t" << y_CQ[i] << "\n";
	}


}

void print(std::complex<double>* D_matrix,double* relative_distance,const double number_rapidity, const int position)
{

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t D_matrix \t position " << position*1.0*LATTICE_SIZE / NX <<"\n";

	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << D_matrix[re].real() << "\n";
	}
}


void print_V(std::complex<double>* V_matrix, double* relative_distance, const double number_rapidity, const int position)
{

	std::ostringstream ofilename_i2;
#ifdef Round_Proton

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N 
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#endif

#else

#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_position_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\V_matrix_position_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";
#endif

#endif

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t V_matrix \t position " << position*1.0*LATTICE_SIZE / NX << "\n";

	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << V_matrix[re].real() << "\n";
	}
}



void print_1step(std::complex<double>* D_matrix, double* relative_distance, const double number_rapidity, const int position,const int number)
{

	std::ostringstream ofilename_i2;
#ifdef Round_Proton


#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_RP_" << position << "_NX_" << NX
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig
		<< "_rapidity_" << number_rapidity << "_number_" << number << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_RP_" << position << "_NX_" << NX
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << "_number_" << number << ".txt";
#endif

#else //Round_Proton
#ifdef v_Parameter
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_" << position << "_NX_" << NX 
		<< "_vPara_" << v_Parameter << "_initial_" << initial_number << "_final_" << number_of_comfig 
		<< "_rapidity_" << number_rapidity << "_number_" << number << ".txt";
#else
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\D_matrix_position_" << position << "_NX_" << NX 
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << "_number_" << number << ".txt";
#endif

#endif // Round_Proton
	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t D_matrix \t position " << position*1.0*LATTICE_SIZE / NX << "\n";

	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << relative_distance[re] << "\t" << D_matrix[re].real() << "\n";
	}
}

//caclulation of the D= 1 - <tr(V^dagger V)/Nc>(|b|= position_B*h,|r|)
void Calculate_D_matrix(int position_B,int maxrap)
{
	double rapidity = 0;
	//int position_B = 6;
	double h = 1.0*LATTICE_SIZE / NX;
	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	std::vector<std::complex<double>> D_matrix(NX / 2, 0);

	//assemble_initial_Quark_position(30);

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0*re*h;
	}

	for (int rap = 0; rap <= maxrap; rap++)
	{
		//rapidity = 2 * rap;
		rapidity = OUTPUT_Y * rap;

		for (int number = initial_number; number < number_of_comfig; number++) {
			Load_matrix_V(V_initial, rapidity, number);



			Eigen::Matrix3cd V_x, V_y, VdV;
			for (int sym_num = 0; sym_num < number_of_symmetry; ++sym_num)
			{
				std::vector<std::complex<double>> D_matrix_temp(NX / 2, 0);
				if (sym_num == 0) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
								V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}
						VdV = V_x.adjoint();
						VdV *= V_y;


						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}

					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}

				}
				else if (sym_num == 1) {
					int posiV =  position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
								V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}

					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 2) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
								V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 3) {
					int posiV = position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
								V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}

			}

		}


		for (int re = 0; re < NX / 2; ++re) {
			D_matrix[re] = D_matrix[re] / ((double)number_of_symmetry) / ((double)(number_of_comfig - initial_number));
		}

		print(D_matrix.data(), relative_distance.data(), rapidity, position_B);
	}

	delete[]V_initial;
}


//caclulation of the D= 1 - <tr(V^dagger V)/Nc>(|b|= position_B*h,|r|)
void Calculate_D_matrix_normal(int position_B, int maxrap)
{
	double rapidity = 0;
	//int position_B = 6;
	double h = 1.0*LATTICE_SIZE / NX;
	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	std::vector<std::complex<double>> D_matrix(NX / 2, 0);

	//assemble_initial_Quark_position(30);

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0*re*h;
	}

	for (int rap = 0; rap <= maxrap; rap++)
	{
		//rapidity = 2 * rap;
		rapidity = OUTPUT_Y * rap;


		Eigen::Matrix3cd A;
		A = Eigen::MatrixXd::Identity(3, 3);

		for (int number = initial_number; number < number_of_comfig; number++) {
			Load_matrix_V(V_initial, rapidity, number);



			Eigen::Matrix3cd V_x, V_y, VdV;
			for (int sym_num = 0; sym_num < number_of_symmetry; ++sym_num)
			{
				std::vector<std::complex<double>> D_matrix_temp(NX / 2, 0);
				if (sym_num == 0) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;


						positionVx = posiV *NX + NX / 2 + vx*NX;
						positionVy = posiV *NX + NX / 2 - vx*NX;

						if (positionVx<0 || positionVx>NX*NX - 1 || positionVy<0 || positionVy>NX*NX - 1) {
							V_x = A;
							V_y = A;
						}
						else {

							for (int i = 0; i < 3; ++i) {
								for (int j = 0; j < 3; ++j) {
									//define V, xi_1, xi_2
									V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
									V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

								}
							}
						}
						VdV = V_x.adjoint();
						VdV *= V_y;


						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}

					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}

				}
				else if (sym_num == 1) {
					int posiV = position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;


						positionVx = posiV *NX + NX / 2 + vx*NX;
						positionVy = posiV *NX + NX / 2 - vx*NX;

						if (positionVx<0 || positionVx>NX*NX - 1 || positionVy<0 || positionVy>NX*NX - 1) {
							V_x = A;
							V_y = A;
						}
						else {

							for (int i = 0; i < 3; ++i) {
								for (int j = 0; j < 3; ++j) {
									//define V, xi_1, xi_2
									V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
									V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

								}
							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}

					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 2) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;

					

						positionVx = posiV + (NX / 2 )*NX + vx;
						positionVy = posiV + (NX / 2 )*NX - vx;

						if (positionVx<0 || positionVx>NX*NX - 1 || positionVy<0 || positionVy>NX*NX - 1) {
							V_x = A;
							V_y = A;
						}
						else {

							for (int i = 0; i < 3; ++i) {
								for (int j = 0; j < 3; ++j) {
									//define V, xi_1, xi_2
									V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
									V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

								}
							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 3) {
					int posiV = position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;

					

						positionVx = posiV + (NX / 2 )*NX + vx;
						positionVy = posiV + (NX / 2 )*NX - vx;

						if (positionVx<0 || positionVx>NX*NX - 1 || positionVy<0 || positionVy>NX*NX - 1) {
							V_x = A;
							V_y = A;
						}
						else {

							for (int i = 0; i < 3; ++i) {
								for (int j = 0; j < 3; ++j) {
									//define V, xi_1, xi_2
									V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
									V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

								}
							}
						}

						VdV = V_x.adjoint();
						VdV *= V_y;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}

			}

		}


		for (int re = 0; re < NX / 2; ++re) {
			D_matrix[re] = D_matrix[re] / ((double)number_of_symmetry) / ((double)(number_of_comfig - initial_number));
		}

		print(D_matrix.data(), relative_distance.data(), rapidity, position_B);
	}

	delete[]V_initial;
}


//caclulation of the D= 1 - <tr(V^dagger V)/Nc>(|b|= position_B*h,|r|)
void Calculate_D_matrix_normal_xdirection(int position_B, int maxrap)
{
	double rapidity = 0;
	//int position_B = 6;
	double h = 1.0*LATTICE_SIZE / NX;
	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	std::vector<std::complex<double>> D_matrix(NX / 2, 0);

	//assemble_initial_Quark_position(30);

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0*re*h;
	}

	for (int rap = 0; rap <= maxrap; rap++)
	{
		//rapidity = 2 * rap;
		rapidity = OUTPUT_Y * rap;


		Eigen::Matrix3cd A;
		A = Eigen::MatrixXd::Identity(3, 3);

		for (int number = initial_number; number < number_of_comfig; number++) {
			Load_matrix_V(V_initial, rapidity, number);



			Eigen::Matrix3cd V_x, V_y, VdV;

				std::vector<std::complex<double>> D_matrix_temp(NX / 2, 0);
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx*NX;
						int positionVy = posiV *NX + NX / 2 - vx*NX;

						if (positionVx<0 || positionVx>NX*NX - 1 || positionVy<0 || positionVy>NX*NX - 1) {
							V_x = A;
							V_y = A;
						}
						else {

							for (int i = 0; i < 3; ++i) {
								for (int j = 0; j < 3; ++j) {
									//define V, xi_1, xi_2
									V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];
									V_y(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

								}
							}
						}
						VdV = V_x.adjoint();
						VdV *= V_y;


						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}

					}

				


		}


		for (int re = 0; re < NX / 2; ++re) {
			D_matrix[re] = D_matrix[re] / ((double)(number_of_comfig - initial_number));
		}

		print(D_matrix.data(), relative_distance.data(), rapidity, position_B);
	}

	delete[]V_initial;
}


//caclulation of the D= 1 - <tr(V)/Nc>(|b|= position_B*h,|r|)
void Calculate_onepoint_matrix(int position_B, int maxrap)
{
	double rapidity = 0;
	//int position_B = 6;
	double h = 1.0*LATTICE_SIZE / NX;
	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	std::vector<std::complex<double>> D_matrix(NX / 2, 0);

	//assemble_initial_Quark_position(30);

	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = 2.0* re*h;
	}

	for (int rap = 0; rap <= maxrap; rap++)
	{
		//rapidity = 2 * rap;
		rapidity = OUTPUT_Y * rap;

		for (int number = initial_number; number < number_of_comfig; number++) {
			Load_matrix_V(V_initial, rapidity, number);



			Eigen::Matrix3cd V_x, V_y, VdV;
			for (int sym_num = 0; sym_num < number_of_symmetry; ++sym_num)
			{
				std::vector<std::complex<double>> D_matrix_temp(NX / 2, 0);
				if (sym_num == 0) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}
						VdV = V_x;
						//VdV *= V_y;


						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							//D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}

					}
					if (number == number_of_comfig - 1) {
						//print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}

				}
				else if (sym_num == 1) {
					int posiV = position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV *NX + NX / 2 + vx;
						int positionVy = posiV *NX + NX / 2 - vx;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];

							}
						}

						VdV = V_x;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							//D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}

					if (number == number_of_comfig - 1) {
						//print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 2) {
					int posiV = -position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVy + 3 * i + j];

							}
						}

						VdV = V_x;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							//D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						//print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}
				else if (sym_num == 3) {
					int posiV = position_B + NX / 2;
					for (int vx = 0; vx < NX / 2; ++vx) {
						int positionVx = posiV + (NX / 2 + vx)*NX;
						int positionVy = posiV + (NX / 2 - vx)*NX;
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								//define V, xi_1, xi_2
								V_x(i, j) = V_initial[3 * 3 * positionVx + 3 * i + j];

							}
						}

						VdV = V_x;
						for (int i = 0; i < 3; ++i) {
							D_matrix[vx] += (1.0 - VdV(i, i)) / ((double)Nc);
							//D_matrix_temp[vx] += (1.0 - VdV(i, i)) / ((double)Nc);

						}
					}
					if (number == number_of_comfig - 1) {
						//print_1step(D_matrix_temp.data(), relative_distance.data(), rapidity, position_B, sym_num);
					}
				}

			}

		}


		for (int re = 0; re < NX / 2; ++re) {
			D_matrix[re] = D_matrix[re] / ((double)number_of_symmetry) / ((double)(number_of_comfig - initial_number));
		}

		print_V(D_matrix.data(), relative_distance.data(), rapidity, position_B);
	}

	delete[]V_initial;
}



void integration_nonElliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_result);

void integration_Elliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_result);

void Integration_Smatrix(std::complex<double>* V_matrix,int max_rap)
{

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}



	//cos_integration(x.data(), y.data());
	for (int rap = 0; rap < max_rap; ++rap) {
		double rapidity = 1.0*rap;
		
	std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 1);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			integration_nonElliptic(V_matrix, integrand_temp1.data());
			for (int n = 0; n<NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }

			integration_Elliptic(V_matrix, integrand_temp2.data());
			for (int n = 0; n<NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }


		}

		for (int n = 0; n<NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		std::ostringstream ofilename_cos;
		ofilename_cos << "test_integration_E_non_E_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_cos(ofilename_cos.str().c_str());

		ofs_res_cos << "#x \t y \t non Elliptic \t Elliptic \n";

		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++)
			{
				ofs_res_cos << x[NX*j + i] << "\t" << y[NX*j + i] << "\t" << integrand_bnonE[NX*j + i].real() << "\t" << integrand_bE[NX*j + i].real() << "\n";
			}
			ofs_res_cos << "\n";
		}
	}

}

void integration_nonElliptic_Wigner(std::complex<double>* V_matrix, std::complex<double>* integrated_result, double momk);

void integration_Elliptic_Wigner(std::complex<double>* V_matrix, std::complex<double>* integrated_result, double momk);

void Integration_Smatrix_towards_Wigner(std::complex<double>* V_matrix, int max_rap)
{

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}



	//cos_integration(x.data(), y.data());
	for (int rap = 0; rap < max_rap; ++rap) {
		double rapidity = 1.0*rap;

		std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 1);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			integration_nonElliptic_Wigner(V_matrix, integrand_temp1.data(),P_UPPER);

			for (int n = 0; n<NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }

			integration_Elliptic_Wigner(V_matrix, integrand_temp2.data(), P_UPPER);
			for (int n = 0; n<NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }


		}

		for (int n = 0; n<NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		std::ostringstream ofilename_cos;
		ofilename_cos << "test_integration_E_non_E_towards_Wigner_NX_"<< NX << "_size_" << LATTICE_SIZE 
			<<"_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_cos(ofilename_cos.str().c_str());

		ofs_res_cos << "#x \t y \t non Elliptic \t Elliptic \n";

		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++)
			{
				ofs_res_cos << x[NX*j + i] << "\t" << y[NX*j + i] << "\t" << integrand_bnonE[NX*j + i].real() << "\t" << integrand_bE[NX*j + i].real() << "\n";
			}
			ofs_res_cos << "\n";
		}
	}

}


void load_integrated_Smatrix(double* I_Smatrix_nonE, double* I_Smatrix_E, const double number_rapidity)
{
	std::ostringstream ifilename;
		ifilename << "test_integration_E_non_E_NX_" << NX << "_size_" << LATTICE_SIZE 
			<< "_rap_" << number_rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ifstream ifs_res(ifilename.str().c_str());

		char str[256];
		if (ifs_res.fail())
		{
			std::cerr << "failed to load file" << std::endl;
		}
		for (int i = 0; i < 4; ++i) {
			ifs_res.getline(str, 256 - 1);
		}
		double x, y;
		for (int j = 0; j < NX; j++) {
			for (int i = 0; i < NX; i++)
			{
				ifs_res >> x  >> y  >> I_Smatrix_nonE[NX*j + i]  >> I_Smatrix_E[NX*j + i] ;
			}
		}
}

void calculate_k_n_4_ncvmp_sp(double* I_Smatrix_nonE, double* I_Smatrix_E, const double Upper_momk,
	double& temp_0_4, double& temp2_0_4, double& temp_0_2, double& temp2_0_2,
	double& temp_2_4, double& temp2_2_4, double& temp_2_2, double& temp2_2_2)
{
	int N = NX;
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0, ymax = h *N / 2.0;
	std::vector<double>   x(N, 0), y(N, 0);
	std::vector<double> IS_nonE(N*N, 0), IS_E(N*N, 0);
	for (int j = 0; j < N; j++) {
		x[j] = xmin + j*h;
		y[j] = ymin + j*h;
		for (int i = 0; i < N; i++)
		{
			IS_nonE[j*N + i] = I_Smatrix_nonE[N*j + i];
			IS_E[N*j + i] = I_Smatrix_E[N*j + i];
		}
	}

	interpolation_2dim IS_nonE_inter, IS_E_inter;
	IS_nonE_inter.set_points(IS_nonE, x, y);
	IS_E_inter.set_points(IS_E, x, y);

	size_t N_mc = MONTE_CARLO_NUMBER;
	// DEVICE: Generate random points distributed in normal distriburion.
	//thrust::host_vector<double> h_random(N_mc), h_random2(N_mc), h_random3(N_mc), h_random4(N_mc),
	//	h_random5(N_mc), h_random6(N_mc), h_random7(N_mc), h_random8(N_mc);
	//thrust::generate(h_random.begin(), h_random.end(), random_point());
	//thrust::generate(h_random2.begin(), h_random2.end(), random_point());
	//thrust::generate(h_random3.begin(), h_random3.end(), random_point());
	//thrust::generate(h_random4.begin(), h_random4.end(), random_point());
	//thrust::generate(h_random.begin(), h_random5.end(), random_point());
	//thrust::generate(h_random2.begin(), h_random6.end(), random_point());
	//thrust::generate(h_random3.begin(), h_random7.end(), random_point());
	//thrust::generate(h_random4.begin(), h_random8.end(), random_point());

	std::random_device seed_gen_1;
	std::default_random_engine engine_1(seed_gen_1());

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

	auto cos_cartesian = [](double x1, double y1, double x2, double y2) {
		if (abs(x1*x1 + y1*y1) < 1e-8 && abs(x2*x2 + y2*y2) <1e-8) {
			return 0.0;
		}
		else if ((x1*x2 + y1*y2) / sqrt(x1*x1 + y1*y1) / sqrt(x2*x2 + y2*y2)>1.0) {
			return 1.0;
		}
		else if ((x1*x2 + y1*y2) / sqrt(x1*x1 + y1*y1) / sqrt(x2*x2 + y2*y2) < -1.0) {
			return -1.0;
		}
		else {
			return (x1*x2 + y1*y2) / sqrt(x1*x1 + y1*y1) / sqrt(x2*x2 + y2*y2);
		}
	};


	double sum_k_0_4 = 0.0;
	double sum_k_2_4 = 0.0;
	double sum2_k_0_4 = 0.0;
	double sum2_k_2_4 = 0.0;
	double sum_k_0_2 = 0.0;
	double sum_k_2_2 = 0.0;
	double sum2_k_0_2 = 0.0;
	double sum2_k_2_2 = 0.0;

#pragma omp parallel for reduction(+:sum_k_0_4,sum_k_2_4,sum2_k_0_4,sum2_k_2_4,sum_k_0_2,sum_k_2_2,sum2_k_0_2,sum2_k_2_2) num_threads(4)
	for (int mcn = 0; mcn < MONTE_CARLO_NUMBER; ++mcn) {
		std::normal_distribution<double> dist_1(0.0, 1.0 / Lamda_QCD_nucleus), dist_2(0.0, 1.0 / Lamda_QCD_nucleus), dist_3(0.0, 1.0 / Lamda_QCD_nucleus),
			dist_4(0.0, 1.0 / Lamda_QCD_nucleus),
			dist_p1(0.0, 1.0 / Lamda_QCD_nucleus), dist_p2(0.0, 1.0 / Lamda_QCD_nucleus), dist_p3(0.0, 1.0 / Lamda_QCD_nucleus),
			dist_p4(0.0, 1.0 / Lamda_QCD_nucleus);
		double b_1, by_1, b_2, by_2, b_3, by_3, b_4, by_4;
		b_1 = dist_1(engine_1);
		by_1 = dist_p1(engine_1);
		b_2 = dist_2(engine_1);
		by_2 = dist_p2(engine_1);
		b_3 = dist_3(engine_1);
		by_3 = dist_p3(engine_1);
		b_4 = dist_4(engine_1);
		by_4 = dist_p4(engine_1);

		double error_temp = 0;

		double phi_1 = arccos_cartesian(b_1, by_1);
		double phi_2 = arccos_cartesian(b_2, by_2);
		double phi_3 = arccos_cartesian(b_3, by_3);
		double phi_4 = arccos_cartesian(b_4, by_4);



		double factor_exp = 1.0 / 4.0*((b_1 - b_2)*(b_1 - b_2) + (by_1 - by_2)*(by_1 - by_2) + (b_1 - b_3)*(b_1 - b_3) + (by_1 - by_3)*(by_1 - by_3)
			+ (b_1 - b_4)*(b_1 - b_4) + (by_1 - by_4)*(by_1 - by_4) + (b_3 - b_2)*(b_3 - b_2) + (by_3 - by_2)*(by_3 - by_2)
			+ (b_4 - b_2)*(b_4 - b_2) + (by_4 - by_2)*(by_4 - by_2) + (b_3 - b_4)*(b_3 - b_4) + (by_3 - by_4)*(by_3 - by_4));
		double factor_exp_k_0_2 = (b_1 - b_2)*(b_1 - b_2) + (by_1 - by_2)*(by_1 - by_2);

		double separate4 = 1.0;
		double separate2 = 1.0;

		//double constant_integration_1 = M_PI * sqrt(M_PI)*exp(- Gauss_param*Gauss_param*Upper_momk*Upper_momk / 8.0)
		//	* boost::math::cyl_bessel_i(0, Gauss_param*Gauss_param*Upper_momk*Upper_momk / 8.0) * sqrt(Gauss_param*Gauss_param*Upper_momk*Upper_momk);


		//double k_func_b_1 = constant_integration_1 - IS_nonE_inter(b_1, by_1);
		//double k_func_b_2 = constant_integration_1 - IS_nonE_inter(b_2, by_2);
		//double k_func_b_3 = constant_integration_1 - IS_nonE_inter(b_3, by_3);
		//double k_func_b_4 = constant_integration_1 - IS_nonE_inter(b_4, by_4);
		double k_func_b_1 = IS_nonE_inter(b_1, by_1);
		double k_func_b_2 = IS_nonE_inter(b_2, by_2);
		double k_func_b_3 = IS_nonE_inter(b_3, by_3);
		double k_func_b_4 = IS_nonE_inter(b_4, by_4);
		//if (b_1 > xmax || by_1 > ymax || b_1 < xmin || by_1 < ymin) {
		//	k_func_b_1 = 
		//}

		double f_A = exp(-Lamda_QCD*Lamda_QCD*factor_exp)
			*separate4
			*k_func_b_1
			*k_func_b_2
			*k_func_b_3
			*k_func_b_4;


		double f_A_k_0_2 = 1.0
			*exp(-Lamda_QCD*Lamda_QCD*factor_exp_k_0_2)
			*separate2
			*k_func_b_1
			*k_func_b_2;


		sum_k_0_2 += f_A_k_0_2;
		sum_k_0_4 += f_A;
		sum2_k_0_2 += f_A_k_0_2*f_A_k_0_2;
		sum2_k_0_4 += f_A*f_A;


		k_func_b_1 = IS_E_inter(b_1, by_1);
		k_func_b_2 = IS_E_inter(b_2, by_2);
		k_func_b_3 = IS_E_inter(b_3, by_3);
		k_func_b_4 = IS_E_inter(b_4, by_4);

		f_A = exp(-Lamda_QCD*Lamda_QCD*factor_exp)
			*cos(2.0*(phi_1 + phi_2 - phi_3 - phi_4))
			*separate4
			*k_func_b_1
			*k_func_b_2
			*k_func_b_3
			*k_func_b_4;


		double f_A_k_2_2 = 1.0
			*exp(-Lamda_QCD*Lamda_QCD*factor_exp_k_0_2)
			*cos(2.0*(phi_1 - phi_2))
			//*(2.0*cos_cartesian(b_1,by_1,b_2,by_2)*cos_cartesian(b_1, by_1, b_2, by_2) - 1.0)
			*separate2
			*k_func_b_1
			*k_func_b_2;

		sum_k_2_2 += f_A_k_2_2;
		sum_k_2_4 += f_A;
		sum2_k_2_2 += f_A_k_2_2*f_A_k_2_2;
		sum2_k_2_4 += f_A*f_A;

	}

	temp_0_2 = sum_k_0_2;
	temp_0_4 = sum_k_0_4;
	temp2_0_2 = sum2_k_0_2;
	temp2_0_4 = sum2_k_0_4;
	temp_2_2 = sum_k_2_2;
	temp_2_4 = sum_k_2_4;
	temp2_2_2 = sum2_k_2_2;
	temp2_2_4 = sum2_k_2_4;
}


void calculate_c_2_4(const int maxrap)
{
	std::vector<double> c_2_2(200, 0);
	std::vector<double> error_c_2_2(200, 0);
	std::vector<double> s_2_4(200, 0);
	std::vector<double> error_s_2_4(200, 0);
	std::vector<double> c_2_4(200, 0);
	std::vector<double> error_c_2_4(200, 0);


	for (int rap = 0; rap < maxrap; ++rap) {
		double rapidity = 1.0*rap;
		std::vector<double> ISmatrix_nonE(NX*NX, 0), ISmatrix_E(NX*NX, 0);
		load_integrated_Smatrix(ISmatrix_nonE.data(), ISmatrix_E.data(), rapidity);

		double sum_k_0_4 = 0.0;
		double sum_k_2_4 = 0.0;
		double sum2_k_0_4 = 0.0;
		double sum2_k_2_4 = 0.0;
		double sum_k_0_2 = 0.0;
		double sum_k_2_2 = 0.0;
		double sum2_k_0_2 = 0.0;
		double sum2_k_2_2 = 0.0;
		calculate_k_n_4_ncvmp_sp(ISmatrix_nonE.data(), ISmatrix_E.data(), P_UPPER, sum_k_0_4, sum2_k_0_4, sum_k_0_2, sum2_k_0_2
			, sum_k_2_4, sum2_k_2_4, sum_k_2_2, sum2_k_2_2);

		double MC_num = MONTE_CARLO_NUMBER;

		c_2_4[rap] = sum_k_2_4 / sum_k_0_4 - 2.0*(sum_k_2_2 / sum_k_0_2)*(sum_k_2_2 / sum_k_0_2);
		double c_2_4_4 = sum_k_2_4 / sum_k_0_4;
		double c_2_4_2 = sum_k_2_2 / sum_k_0_2;
		double error_k_0_4 = sqrt(sum2_k_0_4 / MC_num - sum_k_0_4 / MC_num*sum_k_0_4 / MC_num) / sqrt(MC_num - 1.0);
		double error_k_2_4 = sqrt(sum2_k_2_4 / MC_num - sum_k_2_4 / MC_num*sum_k_2_4 / MC_num) / sqrt(MC_num - 1.0);
		double error_k_0_2 = sqrt(sum2_k_0_2 / MC_num - sum_k_0_2 / MC_num*sum_k_0_2 / MC_num) / sqrt(MC_num - 1.0);
		double error_k_2_2 = sqrt(sum2_k_2_2 / MC_num - sum_k_2_2 / MC_num*sum_k_2_2 / MC_num) / sqrt(MC_num - 1.0);
		double error_4 = MC_num / abs(sum_k_0_4)*sqrt(error_k_2_4*error_k_2_4 + c_2_4_4 * c_2_4_4 * error_k_0_4*error_k_0_4);
		double error_2 = MC_num / abs(sum_k_0_2)*sqrt(error_k_2_2*error_k_2_2 + c_2_4_2 * c_2_4_2 * error_k_0_2*error_k_0_2);

		c_2_2[rap] = c_2_4_2;
		error_c_2_2[rap] = error_2;
		s_2_4[rap] = c_2_4_4;
		error_s_2_4[rap] = error_4;
		error_c_2_4[rap] = sqrt(error_4*error_4 + 4.0*2.0*error_2*error_2*c_2_4_2*c_2_4_2);


	}
	double epoch = 1;
	std::ostringstream ofilename;
	ofilename << "c_2_4_k_spline_initc_"<< (number_of_comfig - initial_number) <<"_N_" << MONTE_CARLO_NUMBER << "_epoch_" << epoch << ".txt";
	std::ofstream ofs_res(ofilename.str().c_str());

	ofs_res << "#rap \t c_2_4 \t error \t s_2_4 \t error \t c_2_2 \t error \n";

	for (int i = 0; i < maxrap; ++i) {
		ofs_res << 1.0*i << "\t" << c_2_4[i] << "\t" << error_c_2_4[i] << "\t" << s_2_4[i] << "\t" << error_s_2_4[i] << "\t" << c_2_2[i] << "\t" << error_c_2_2[i] << "\n";
	}
}


//generate unitary matrixes inside the square and unit matrixes at the outside.
void Initialize_unit_matrix(std::complex<double>* V_initial)
{
	Eigen::Matrix3cd A, Unitary_example;
	A = Eigen::MatrixXd::Identity(3, 3);
	//tr(Unitary_example) = -1, Unitary_example*Unitary_example^dagger = Identity
	Unitary_example << std::complex<double>(-1.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
		std::complex<double>(0.0, 0.0), std::complex<double>(-1.0, 0.0), std::complex<double>(0.0, 0.0),
		std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0);


	for (int ix = 0; ix < NX; ++ix) {
		for (int jy = 0; jy < NX; ++jy) {


				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						V_initial[3 * 3 * (NX*ix + jy) + 3 * i + j] = A(i, j);

					}
				}

		}

	}

}


void Smatrix_value(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k);

void Derive_Wigner_distribution_revised(std::complex<double>* V_matrix, int max_rap, int num_mom)
{
	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 1.0*rap;

		int num_bspace = 0;
		std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2,0), WignerS(num_mom*NX / 2,0), EWignerS(num_mom*NX / 2, 0);

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;


			std::vector<double> b_positionS(NX/2, 0), b_integrated_valueS(NX/2, 0), b_integrated_EvalueS(NX/2, 0);

			std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 0);
			for (int num = initial_number; num < number_of_comfig; ++num) {
				int number = num;
				Load_matrix_V(V_matrix, rapidity, number);
				//integration_nonElliptic_Wigner(V_matrix, integrand_temp1.data(), momk);
				Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
				for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }

				//integration_Elliptic_Wigner(V_matrix, integrand_temp2.data(), momk);
				for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }


			}

			for (int n = 0; n < NX*NX; ++n) {
				integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
				integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
			}



			double h = 1.0*LATTICE_SIZE / NX;

			std::vector<double> b_position(1, 0), b_integrated_value(1, 0), b_integrated_Evalue(1, 0);
			double impact_paramb = 0;
			double integrated = 0;
			double integratedE = 0;
			//x axis
			for (int i = 1; i <= NX/2; i++) {

				std::vector<double> temp_vector, temp_integrated, temp_integrated_E;
				impact_paramb = abs((1.0*i - NX / 2.0 ))*h*sqrt(2.0);
				integrated = (integrand_bnonE[NX*i + i].real() + integrand_bnonE[NX*(NX  - i) + NX  - i].real()
					+ integrand_bnonE[NX*i + NX - i].real() + integrand_bnonE[NX*(NX  - i) + i].real()) / 4.0;
				integratedE = (integrand_bE[NX*i + i].real() + integrand_bE[NX*(NX  - i) + NX  - i].real()
					+ integrand_bE[NX*i + NX  - i].real() + integrand_bE[NX*(NX  - i) + i].real()) / 4.0;

					//b_position.push_back(impact_paramb);
					//b_integrated_value.push_back(integrated);
					//b_integrated_Evalue.push_back(integratedE);

					b_positionS[NX/2 - i ] = impact_paramb;
					b_integrated_valueS[NX/2 - i] = integrated;
					b_integrated_EvalueS[NX/2 - i ] = integratedE;

				//for (int j = 0; j < i; j++) {


				//	std::vector<double> temp_vector, temp_integrated;
				//	impact_paramb = abs((1.0*i - NX / 2.0 + 1.0 / 2.0))*h*sqrt(2.0);
				//	integrated = (integrand_bnonE[NX*j + i].real() + integrand_bnonE[NX*(NX - 1 - j) + NX - 1 - i].real()
				//		+ integrand_bnonE[NX*j + NX - 1 - i].real() + integrand_bnonE[NX*(NX - 1 - j) + i].real()) / 4.0
				//		+ (integrand_bnonE[NX*i + j].real() + integrand_bnonE[NX*(NX - 1 - i) + NX - 1 - j].real()
				//			+ integrand_bnonE[NX*i + NX - 1 - j].real() + integrand_bnonE[NX*(NX - 1 - i) + j].real()) / 4.0;
				//	integratedE = (integrand_bE[NX*j + i].real() + integrand_bE[NX*(NX - 1 - j) + NX - 1 - i].real()
				//		+ integrand_bE[NX*j + NX - 1 - i].real() + integrand_bE[NX*(NX - 1 - j) + i].real()) / 4.0
				//		+ (integrand_bE[NX*i + j].real() + integrand_bE[NX*(NX - 1 - i) + NX - 1 - j].real()
				//			+ integrand_bE[NX*i + NX - 1 - j].real() + integrand_bE[NX*(NX - 1 - i) + j].real()) / 4.0;

						//b_position.push_back(impact_paramb);
						//b_integrated_value.push_back(integrated);
						//b_integrated_Evalue.push_back(integratedE);

				//}
			}

			num_bspace = int(b_position.begin() - b_position.end());
			b_space = b_position;

			b_spaceS = b_positionS;


			//The spline interpolation of the integrated using  Tino Kluge code.
			//tk::spline b_integrated_value_spline, b_integrated_Evalue_spline;
			//b_integrated_value_spline.set_points(b_position, b_integrated_value);
			//b_integrated_Evalue_spline.set_points(b_position, b_integrated_Evalue);

			tk::spline b_integrated_value_splineS, b_integrated_Evalue_splineS;
			b_integrated_value_splineS.set_points(b_positionS, b_integrated_valueS);
			b_integrated_Evalue_splineS.set_points(b_positionS, b_integrated_EvalueS);

			double coefficient = Nc / 2.0 / ALPHA_S / M_PI / M_PI;

			//impact_param = 0 -> W = infty?
			for (int i = 1; i < NX / 2; i++) {

				WignerS[NX / 2 *mom + i] = coefficient*(1.0 / 4.0*b_integrated_value_splineS.Sec_deriv(b_positionS[i])
					+ 1.0 / 4.0 / b_positionS[i] * b_integrated_value_splineS.Fst_deriv(b_positionS[i])
					+ momk*momk*b_integrated_value_splineS(b_positionS[i]));

				EWignerS[NX / 2 *mom + i] = -coefficient*(1.0 / 4.0*b_integrated_Evalue_splineS.Sec_deriv(b_positionS[i])
					+ 1.0 / 4.0 / b_positionS[i] * b_integrated_Evalue_splineS.Fst_deriv(b_positionS[i])
					- 1.0 / b_positionS[i] / b_positionS[i] * b_integrated_Evalue_splineS(b_positionS[i])
					+ momk*momk*b_integrated_Evalue_splineS(b_positionS[i]));
			}


		}


		std::ostringstream ofilename_Wigner;
		ofilename_Wigner << "E_non_E_towards_Wigner_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

		ofs_res_Wigner << "#b \t momk \t non Elliptic \t Elliptic \n";

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;

			//impact_param = 0 -> W = infty?
			for (int j = 1; j < NX / 2; j++) {

				int jkai = j;
				ofs_res_Wigner << b_spaceS[jkai] << "\t" << momk << "\t" << WignerS[NX / 2 *mom + jkai] << "\t" << EWignerS[NX / 2 *mom + jkai] << "\n";
			}
			ofs_res_Wigner << "\n";
		}
	}

}


void Derive_Wigner_distribution_fromSmatrix(std::complex<double>* V_matrix, int max_rap, int num_mom)
{
	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;

		int num_bspace = 0;
		std::vector<double> b_space(NX / 2, 0), Wigner(num_mom*NX / 2, 0), EWigner(num_mom*NX / 2, 0),
			Smat(num_mom*NX / 2, 0), SmatFd(num_mom*NX / 2, 0), SmatSd(num_mom*NX / 2, 0),
			b_spaceS(NX / 2, 0), WignerS(num_mom*NX / 2, 0), EWignerS(num_mom*NX / 2, 0);

		std::ostringstream ifilename_Wigner;
		ifilename_Wigner << "S_Wigner_diag_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ifstream ifs_res_Wigner(ifilename_Wigner.str().c_str());

		char str[256];

		if (ifs_res_Wigner.fail())
		{
			std::cerr << "failed to load file "<< "\t" << rapidity << std::endl;
		}
		ifs_res_Wigner.getline(str, 256 - 1);

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;
			double momentum = 0;
			//impact_param = 0 -> W = infty?
			for (int j = 0; j < NX / 2; j++) {

				int jkai = j;
				ifs_res_Wigner >> b_space[jkai] >> momentum >> Wigner[NX / 2 * mom + jkai] >> EWigner[NX / 2 * mom + jkai];
			}
		}

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;


			std::vector<double> b_positionS(NX / 2, 0), b_integrated_valueS(NX / 2, 0), b_integrated_EvalueS(NX / 2, 0);

			std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 0);
		
			double h = 1.0*LATTICE_SIZE / NX;

			std::vector<double> b_position(1, 0), b_integrated_value(1, 0), b_integrated_Evalue(1, 0);
			double impact_paramb = 0;
			double integrated = 0;
			double integratedE = 0;
			//x axis
			for (int i = 0; i < NX / 2; i++) {

				b_positionS[i] = b_space[i];
				b_integrated_valueS[i] = Wigner[NX / 2 * mom + i];
				b_integrated_EvalueS[i]  = EWigner[NX / 2 * mom + i];

			}


			//The spline interpolation of the integrated using  Tino Kluge code.
			//tk::spline b_integrated_value_spline, b_integrated_Evalue_spline;
			//b_integrated_value_spline.set_points(b_position, b_integrated_value);
			//b_integrated_Evalue_spline.set_points(b_position, b_integrated_Evalue);

			tk::spline b_integrated_value_splineS, b_integrated_Evalue_splineS;
			b_integrated_value_splineS.set_points(b_space, b_integrated_valueS);
			b_integrated_Evalue_splineS.set_points(b_space, b_integrated_EvalueS);

			double coefficient = 1.0;

			//impact_param = 0 -> W = infty?
			for (int i = 1; i < NX / 2; i++) {

				WignerS[NX / 2 * mom + i] = coefficient*(1.0 / 4.0*b_integrated_value_splineS.Sec_deriv(b_space[i])
					+ 1.0 / 4.0 / b_space[i] * b_integrated_value_splineS.Fst_deriv(b_space[i])
					+ momk*momk*b_integrated_value_splineS(b_space[i]));

				//EWignerS[NX / 2 * mom + i] = -coefficient*(1.0 / 4.0*b_integrated_Evalue_splineS.Sec_deriv(b_space[i])
				//	+ 1.0 / 4.0 / b_positionS[i] * b_integrated_Evalue_splineS.Fst_deriv(b_space[i])
				//	- 1.0 / b_space[i] / b_space[i] * b_integrated_Evalue_splineS(b_space[i])
				//	+ momk*momk*b_integrated_Evalue_splineS(b_space[i]));
				Smat[NX / 2 * mom + i] = b_integrated_value_splineS(b_space[i]);
				SmatFd[NX / 2 * mom + i] = b_integrated_value_splineS.Fst_deriv(b_space[i]);
				SmatSd[NX / 2 * mom + i] = b_integrated_value_splineS.Sec_deriv(b_space[i]);

			}


		}


		std::ostringstream ofilename_Wigner;
		ofilename_Wigner << "E_non_E_Wigner_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

		ofs_res_Wigner << "#b \t momk \t non Elliptic \t Smat \t Smat Fderiv \t Smat Sderiv \n";

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;

			//impact_param = 0 -> W = infty?
			for (int j = 1; j < NX / 2; j++) {

				int jkai = j;
				ofs_res_Wigner << b_space[jkai] << "\t" << momk << "\t" << WignerS[NX / 2 * mom + jkai] << "\t" << Smat[NX / 2 * mom + jkai] 
					<< "\t" << SmatFd[NX / 2 * mom + jkai] << "\t" << SmatSd[NX / 2 * mom + jkai] << "\n";
			}
			ofs_res_Wigner << "\n";
		}
	}

}


void nonElliptic(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDP, std::complex<double>* integrated_resultWW, double mom_k);

void Derive_Wigner_distribution_DP_WW(std::complex<double>* V_matrix, int max_rap, int num_mom)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 

	for (int rap = 0; rap < max_rap; ++rap) {
		double rapidity = 1.0*rap;

		int num_bspace = 0;
		std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2, 0), WignerS(num_mom*NX / 2, 0), EWignerS(num_mom*NX / 2, 0);

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;


			std::vector<double> b_positionS(NX / 2, 0), b_integrated_valueS(NX / 2, 0), b_integrated_EvalueS(NX / 2, 0);

			std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 0);
			for (int num = initial_number; num < number_of_comfig; ++num) {
				int number = num;
				Load_matrix_V(V_matrix, rapidity, number);
				//Initialize_unit_matrix(V_matrix);
				nonElliptic(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
				for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }

				for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }


			}

			for (int n = 0; n < NX*NX; ++n) {
				integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
				integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
			}



			double h = 1.0*LATTICE_SIZE / NX;

			std::vector<double> b_position(NX / 2, 0), b_integrated_value(NX / 2, 0), b_integrated_Evalue(NX / 2, 0);
			double impact_paramb = 0;
			double integrated = 0;
			double integratedE = 0;
			//x axis
			for (int i = 0; i < NX / 2; i++) {

				impact_paramb = abs((1.0*i - NX / 2.0 + 1.0 / 2.0))*h*sqrt(2.0);
				integrated = (integrand_bnonE[NX*i + i].real() + integrand_bnonE[NX*(NX - 1 - i) + NX - 1 - i].real()
					+ integrand_bnonE[NX*i + NX - 1 - i].real() + integrand_bnonE[NX*(NX - 1 - i) + i].real()) / 4.0;
				integratedE = (integrand_bE[NX*i + i].real() + integrand_bE[NX*(NX - 1 - i) + NX - 1 - i].real()
					+ integrand_bE[NX*i + NX - 1 - i].real() + integrand_bE[NX*(NX - 1 - i) + i].real()) / 4.0;


				b_positionS[NX / 2 - i - 1] = impact_paramb;
				b_integrated_valueS[NX / 2 - i - 1] = integrated;
				b_integrated_EvalueS[NX / 2 - i - 1] = integratedE;

				for (int j = 0; j < i; j++) {


					impact_paramb = abs((1.0*i - NX / 2.0 + 1.0 / 2.0))*h*sqrt(2.0);
					integrated = (integrand_bnonE[NX*j + i].real() + integrand_bnonE[NX*(NX - 1 - j) + NX - 1 - i].real()
						+ integrand_bnonE[NX*j + NX - 1 - i].real() + integrand_bnonE[NX*(NX - 1 - j) + i].real()) / 4.0
						+ (integrand_bnonE[NX*i + j].real() + integrand_bnonE[NX*(NX - 1 - i) + NX - 1 - j].real()
							+ integrand_bnonE[NX*i + NX - 1 - j].real() + integrand_bnonE[NX*(NX - 1 - i) + j].real()) / 4.0;
					integratedE = (integrand_bE[NX*j + i].real() + integrand_bE[NX*(NX - 1 - j) + NX - 1 - i].real()
						+ integrand_bE[NX*j + NX - 1 - i].real() + integrand_bE[NX*(NX - 1 - j) + i].real()) / 4.0
						+ (integrand_bE[NX*i + j].real() + integrand_bE[NX*(NX - 1 - i) + NX - 1 - j].real()
							+ integrand_bE[NX*i + NX - 1 - j].real() + integrand_bE[NX*(NX - 1 - i) + j].real()) / 4.0;

					//b_position.push_back(impact_paramb);
					//b_integrated_value.push_back(integrated);
					//b_integrated_Evalue.push_back(integratedE);

				}
			}

			num_bspace = int(b_position.begin() - b_position.end());
			b_space = b_position;

			b_spaceS = b_positionS;


			for (int i = 0; i < NX / 2; i++) {

				WignerS[num_mom*mom + i] = b_integrated_valueS[i];

				EWignerS[num_mom*mom + i] = b_integrated_EvalueS[i];
			}


		}


		std::ostringstream ofilename_Wigner;
		ofilename_Wigner << "DPE_WWEE_Wigner_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

		ofs_res_Wigner << "#b \t momk \t DP \t WW \n";

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;
			for (int j = 0; j < NX / 2; j++) {

				ofs_res_Wigner << b_spaceS[j] << "\t" << momk << "\t" << WignerS[num_mom*mom + j] << "\t" << EWignerS[num_mom*mom + j] << "\n";
			}
			ofs_res_Wigner << "\n";
		}
	}

}


void Derive_Wigner_distribution_DP_WW_diagonal(std::complex<double>* V_matrix, int max_rap, int num_mom)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 

	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;

		int num_bspace = 0;
		std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2, 0), WignerS(num_mom*NX / 2, 0), EWignerS(num_mom*NX / 2, 0);

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;


			std::vector<double> b_positionS(NX / 2, 0), b_integrated_valueS(NX / 2, 0), b_integrated_EvalueS(NX / 2, 0);

			std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 2);
			for (int num = initial_number; num < number_of_comfig; ++num) {
				int number = num;
				Load_matrix_V(V_matrix, rapidity, number);
				//Initialize_unit_matrix(V_matrix);
				nonElliptic(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
				//Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
				for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }

				for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }


			}

			for (int n = 0; n < NX*NX; ++n) {
				integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
				integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
			}



			double h = 1.0*LATTICE_SIZE / NX;

			std::vector<double> b_position(NX / 2, 0), b_integrated_value(NX / 2, 0), b_integrated_Evalue(NX / 2, 0);
			double impact_paramb = 0;
			double integrated = 0;
			double integratedE = 0;
			//x axis
			for (int i = 1; i <= NX / 2; i++) {

				impact_paramb = abs((1.0*i - NX / 2.0 ))*h*sqrt(2.0);
				integrated = (integrand_bnonE[NX*i + i].real() + integrand_bnonE[NX*(NX  - i) + NX  - i].real()
					+ integrand_bnonE[NX*i + NX  - i].real() + integrand_bnonE[NX*(NX - i) + i].real()) / 4.0;
				integratedE = (integrand_bE[NX*i + i].real() + integrand_bE[NX*(NX - i) + NX  - i].real()
					+ integrand_bE[NX*i + NX - i].real() + integrand_bE[NX*(NX - i) + i].real()) / 4.0;


				//b_positionS[NX / 2 - i ] = impact_paramb;
				//b_integrated_valueS[NX / 2 - i ] = integrated;
				//b_integrated_EvalueS[NX / 2 - i ] = integratedE;

				b_positionS[NX / 2 - i] = impact_paramb;
				b_integrated_valueS[NX / 2 - i] = integrand_bnonE[NX*i + i].real();
				b_integrated_EvalueS[NX / 2 - i] = integrand_bE[NX*i + i].real();

			}

			num_bspace = int(b_position.begin() - b_position.end());
			b_space = b_position;

			b_spaceS = b_positionS;

			for (int i = 0; i < NX / 2; i++) {

				WignerS[NX / 2 *mom + i] = b_integrated_valueS[i];

				EWignerS[NX / 2 *mom + i] = b_integrated_EvalueS[i];
			}


#ifndef WIGNER
			int N = NX;
			double   xmax = h *N / 2.0, xmin = -h*N / 2.0, ymin = -h*N / 2.0,
				s = 0.1, s2 = s*s;
			double   *x = new double[N*N], *y = new double[N*N];
			double r2;
			for (int j = 0; j < N; j++) {
				for (int i = 0; i < N; i++)
				{
					x[N*j + i] = xmin + i*h;
					y[N*j + i] = ymin + j*h;
					//f[N*j + i] = ;
				}
			}

			std::ostringstream ofilename_Wigner_all;
			ofilename_Wigner_all << "UsubUdagger_unitary_NX_" << NX << "_size_" << LATTICE_SIZE
				<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
			std::ofstream ofs_res_Wigner_all(ofilename_Wigner_all.str().c_str());
			ofs_res_Wigner_all << "#b_x \t b_y \t DP \t WW \n";

			for (int mom = 0; mom < NX; mom++) {
				for (int j = 0; j < NX; j++) {
					int index = j*NX + mom;
					ofs_res_Wigner_all << x[index] << "\t" << y[index] << "\t" << abs(integrand_bnonE[index]) << "\t" << abs(integrand_bE[index]) << "\n";
				}
				ofs_res_Wigner_all << "\n";
			}
#endif


			std::cout<< "mom" << mom << "\n";
		}


		std::ostringstream ofilename_Wigner, ofilename_Wigner_all;
		ofilename_Wigner << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\DPE_WWEE_Wigner_diag_woaa_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

		ofs_res_Wigner << "#b \t momk \t DP \t WW \n";

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;
			for (int j = 0; j < NX / 2; j++) {

				ofs_res_Wigner << b_spaceS[j] << "\t" << momk << "\t" << WignerS[NX / 2 *mom + j] << "\t" << EWignerS[NX / 2 *mom + j] << "\n";
			}
			ofs_res_Wigner << "\n";
		}

		

	}

}


void TMD_value(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP);

void Derive_TMD_DP(std::complex<double>* V_matrix, int max_rap)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 
	double   *k = new double[NX];
	for (int i = 0; i <= NX / 2; i++)
	{
		k[i] = i * 2.0 * M_PI / LATTICE_SIZE;
	}
	for (int i = NX / 2 + 1; i < NX; i++)
	{
		k[i] = (i - NX) * 2.0 * M_PI / LATTICE_SIZE;
	}

	for (int rap = 0; rap <= max_rap; ++rap) {
		//double rapidity = 2.0*rap + 4.0;
		double rapidity = 5.0*rap;


		std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 2);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			//Initialize_unit_matrix(V_matrix);
			TMD_value(V_matrix, integrand_temp1.data(), integrand_temp2.data());

			std::cout << num << "\n";

			//Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }
		}

#pragma omp parallel for num_threads(6)
		for (int n = 0; n < NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		//we need (L/N)^2*(L/N)^2 <- |\int d^2x e^ikx|^2
		double coeff = 1.0 / ALPHA_S / 2.0 / M_PI / M_PI*((double)LATTICE_SIZE)/((double)NX)*((double)LATTICE_SIZE) / ((double)NX)
			*((double)LATTICE_SIZE) / ((double)NX)*((double)LATTICE_SIZE) / ((double)NX);

		std::ostringstream ofilename_Wigner,diagonal;
		ofilename_Wigner << "nonnormalized_DP_TMD_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		diagonal << "nonnormalized_DP_TMD_diagonal_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());
		std::ofstream ofs_res_diagonal(diagonal.str().c_str());

		ofs_res_Wigner << "#kx \t ky \t DPk \t DP \t (DPk-DP)/DPk \n";
		ofs_res_diagonal << "#k \t DPk \t DP \t (DPk-DP)/DPk \n";

		for (int i = NX / 2 + 1; i < NX; i++) {
			for (int j = NX / 2 + 1; j < NX; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real() )/ integrand_bnonE[NX *i + j].real() << "\n";
			}

			for (int j = 0; j <= NX / 2; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			ofs_res_Wigner << "\n";
		}

		for (int i = 0; i <= NX / 2; i++) {
			for (int j = NX / 2 + 1; j < NX; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			for (int j = 0; j <= NX / 2; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			ofs_res_Wigner << "\n";
		}

		for (int i = 0; i < NX / 2; i++) {
			ofs_res_diagonal << sqrt(2.0)*k[i] << "\t" << coeff*integrand_bnonE[NX *i + i].real() << "\t" << coeff*integrand_bE[NX *i + i].real()
				<< "\t" << (integrand_bnonE[NX *i + i].real() - integrand_bE[NX *i + i].real()) / integrand_bnonE[NX *i + i].real() << "\n";
		}
		

	}
	delete k;
}

void TMD_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP);

void Derive_TMD_DP_direct(std::complex<double>* V_matrix, int max_rap)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 
	double   *k = new double[NX];
	for (int i = 0; i < NX ; i++)
	{
		k[i] = (i - NX / 2)*2.0*M_PI/1.0 / LATTICE_SIZE;
	}

	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;


		std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 2);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			//Initialize_unit_matrix(V_matrix);
			TMD_direct(V_matrix, integrand_temp1.data(), integrand_temp2.data());

			std::cout << num << "\n";

			//Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }
		}

#pragma omp parallel for num_threads(6)
		for (int n = 0; n < NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		double coeff = 1.0 / ALPHA_S / 2.0 / M_PI / M_PI;

		std::ostringstream ofilename_Wigner, diagonal;
		ofilename_Wigner << "nonnormalized_DP_TMD_direct_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		diagonal << "nonnormalized_DP_TMD_direct_diagonal_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());
		std::ofstream ofs_res_diagonal(diagonal.str().c_str());

		ofs_res_Wigner << "#kx \t ky \t DPk \t DP \t (DPk-DP)/DPk \n";
		ofs_res_diagonal << "#k \t DPk \t DP \t (DPk-DP)/DPk \n";


		for (int i = 0; i < NX; i++) {

			for (int j = 0; j < NX ; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			ofs_res_Wigner << "\n";
		}


		for (int i = NX / 2; i < NX; i++) {
			ofs_res_diagonal << sqrt(2.0)*k[i] << "\t" << coeff*integrand_bnonE[NX *i + i].real() << "\t" << coeff*integrand_bE[NX *i + i].real()
				<< "\t" << (integrand_bnonE[NX *i + i].real() - integrand_bE[NX *i + i].real()) / integrand_bnonE[NX *i + i].real() << "\n";
		}

	}
	delete k;
}


void Wigner_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP, double momk);

void Wigner_DP_WW_diagonal(std::complex<double>* V_matrix, int max_rap, int num_mom)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 

	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;

		int num_bspace = 0;
		std::vector<double> b_space, Wigner, EWigner, b_spaceS(NX / 2, 0), WignerS(num_mom*NX / 2, 0), EWignerS(num_mom*NX / 2, 0);

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;


			std::vector<double> b_positionS(NX / 2, 0), b_integrated_valueS(NX / 2, 0), b_integrated_EvalueS(NX / 2, 0);

			std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 2);
			for (int num = initial_number; num < number_of_comfig; ++num) {
				int number = num;
				Load_matrix_V(V_matrix, rapidity, number);
				//Initialize_unit_matrix(V_matrix);
				Wigner_direct(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
				//Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);

#pragma omp parallel for num_threads(6)
				for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }
#pragma omp parallel for num_threads(6)
				for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }

			}

#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) {
				integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
				integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
			}



			double h = 1.0*LATTICE_SIZE / NX;

			std::vector<double> b_position(NX / 2, 0), b_integrated_value(NX / 2, 0), b_integrated_Evalue(NX / 2, 0);
			double impact_paramb = 0;
			double integrated = 0;
			double integratedE = 0;
			//x axis
			for (int i = 1; i <= NX / 2; i++) {

				impact_paramb = abs((1.0*i - NX / 2.0))*h*sqrt(2.0);
				integrated = (integrand_bnonE[NX*i + i].real() + integrand_bnonE[NX*(NX - i) + NX - i].real()
					+ integrand_bnonE[NX*i + NX - i].real() + integrand_bnonE[NX*(NX - i) + i].real()) / 4.0;
				integratedE = (integrand_bE[NX*i + i].real() + integrand_bE[NX*(NX - i) + NX - i].real()
					+ integrand_bE[NX*i + NX - i].real() + integrand_bE[NX*(NX - i) + i].real()) / 4.0;


				//b_positionS[NX / 2 - i ] = impact_paramb;
				//b_integrated_valueS[NX / 2 - i ] = integrated;
				//b_integrated_EvalueS[NX / 2 - i ] = integratedE;

				b_positionS[NX / 2 - i] = impact_paramb;
				b_integrated_valueS[NX / 2 - i] = integrand_bnonE[NX*i + i].real();
				b_integrated_EvalueS[NX / 2 - i] = integrand_bE[NX*i + i].real();

			}

			num_bspace = int(b_position.begin() - b_position.end());
			b_space = b_position;

			b_spaceS = b_positionS;

			for (int i = 0; i < NX / 2; i++) {

				WignerS[NX / 2 * mom + i] = b_integrated_valueS[i];

				EWignerS[NX / 2 * mom + i] = b_integrated_EvalueS[i];
			}


			std::cout << "mom" << mom << "\n";
		}


		std::ostringstream ofilename_Wigner, ofilename_Wigner_all;
		ofilename_Wigner << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\DP_WW_Wigner_diag_direct_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());

		ofs_res_Wigner << "#b \t momk \t DP \t WW \n";

		for (int mom = 0; mom < num_mom; mom++) {
			double momk = P_UPPER / num_mom*mom;
			for (int j = 0; j < NX / 2; j++) {

				ofs_res_Wigner << b_spaceS[j] << "\t" << momk << "\t" << WignerS[NX / 2 * mom + j] << "\t" << EWignerS[NX / 2 * mom + j] << "\n";
			}
			ofs_res_Wigner << "\n";
		}



	}

}

//only produce \prtial_i U^\dagger \partial^i U
void GPD_direct(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP);

//using integration formula of the above quantity to get the GPD
void GPD_FFT(std::complex<double>* V_matrix, std::complex<double>* integrated_resultDPk, std::complex<double>* integrated_resultDP);

void Derive_GPD_direct(std::complex<double>* V_matrix, int max_rap)
{
	//notice!! Wigner:DP, EWigner:WW 

	double   *x = new double[NX*NX], *y = new double[NX*NX];
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *NX / 1.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;

	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;


		std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 0), integrand_temp2(NX*NX, 0);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			//Initialize_unit_matrix(V_matrix);
			//GPD_direct(V_matrix, integrand_temp1.data(), integrand_temp2.data());
			GPD_FFT(V_matrix, integrand_temp1.data(), integrand_temp2.data());
			std::cout << num << "\n";

#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }
		}

#pragma omp parallel for num_threads(6)
		for (int n = 0; n < NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		double coeff = 1.0 / ALPHA_S/2.0/M_PI/M_PI;

		std::ostringstream ofilename_Wigner, diagonal;
		ofilename_Wigner << "GPD_direct_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		diagonal << "GPD_direct_diagonal_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());
		std::ofstream ofs_res_diagonal(diagonal.str().c_str());

		ofs_res_Wigner << "#bx \t by \t DP \t WW \t (DP-WW)/DP \n";
		ofs_res_diagonal << "#b \t DP \t WW \t (DP-WW)/DP \n";


		for (int i = 0; i < NX; i++) {

			for (int j = 0; j < NX; j++) {
				ofs_res_Wigner << x[NX*i + j] << "\t" << y[NX*i + j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			ofs_res_Wigner << "\n";
		}


		for (int i = NX / 2; i < NX; i++) {
			ofs_res_diagonal << sqrt(2.0)*x[NX*i + i] << "\t" << coeff*integrand_bnonE[NX *i + i].real() << "\t" << coeff*integrand_bE[NX *i + i].real()
				<< "\t" << (integrand_bnonE[NX *i + i].real() - integrand_bE[NX *i + i].real()) / integrand_bnonE[NX *i + i].real() << "\n";
		}

	}
	delete[]x;
	delete[]y;
}


void Derive_Dmatrix(std::complex<double>* V_matrix, int max_rap)
{
	//notice!! Wigner:DPWigner, EWigner:WWWigner 
	double   *k = new double[NX];
	for (int i = 0; i < NX; i++)
	{
		k[i] = (i - NX / 2)*2.0*M_PI / 1.0 / LATTICE_SIZE;
	}

	for (int rap = 0; rap <= max_rap; ++rap) {
		double rapidity = 5.0*rap;


		std::vector<std::complex<double>> integrand_bnonE(NX*NX, 0), integrand_bE(NX*NX, 0), integrand_temp1(NX*NX, 1), integrand_temp2(NX*NX, 2);
		for (int num = initial_number; num < number_of_comfig; ++num) {
			int number = num;
			Load_matrix_V(V_matrix, rapidity, number);
			//Initialize_unit_matrix(V_matrix);
			TMD_direct(V_matrix, integrand_temp1.data(), integrand_temp2.data());

			std::cout << num << "\n";

			//Smatrix_value(V_matrix, integrand_temp1.data(), integrand_temp2.data(), momk);
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bnonE[n] += integrand_temp1[n]; }
#pragma omp parallel for num_threads(6)
			for (int n = 0; n < NX*NX; ++n) { integrand_bE[n] += integrand_temp2[n]; }
		}

#pragma omp parallel for num_threads(6)
		for (int n = 0; n < NX*NX; ++n) {
			integrand_bnonE[n] = integrand_bnonE[n] / ((double)(number_of_comfig - initial_number));
			integrand_bE[n] = integrand_bE[n] / ((double)(number_of_comfig - initial_number));
		}

		double coeff = 1.0 / ALPHA_S / 2.0 / M_PI / M_PI;

		std::ostringstream ofilename_Wigner, diagonal;
		ofilename_Wigner << "Dmatrix_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		diagonal << "Dmatrix_diagonal_NX_" << NX << "_size_" << LATTICE_SIZE
			<< "_rap_" << rapidity << "_config_" << (number_of_comfig - initial_number) << "_real.txt";
		std::ofstream ofs_res_Wigner(ofilename_Wigner.str().c_str());
		std::ofstream ofs_res_diagonal(diagonal.str().c_str());

		ofs_res_Wigner << "#kx \t ky \t DPk \t DP \t (DPk-DP)/DPk \n";
		ofs_res_diagonal << "#k \t DPk \t DP \t (DPk-DP)/DPk \n";


		for (int i = 0; i < NX; i++) {

			for (int j = 0; j < NX; j++) {
				ofs_res_Wigner << k[i] << "\t" << k[j] << "\t" << coeff*integrand_bnonE[NX *i + j].real() << "\t" << coeff*integrand_bE[NX *i + j].real()
					<< "\t" << (integrand_bnonE[NX *i + j].real() - integrand_bE[NX *i + j].real()) / integrand_bnonE[NX *i + j].real() << "\n";
			}

			ofs_res_Wigner << "\n";
		}


		for (int i = NX / 2; i < NX; i++) {
			ofs_res_diagonal << sqrt(2.0)*k[i] << "\t" << coeff*integrand_bnonE[NX *i + i].real() << "\t" << coeff*integrand_bE[NX *i + i].real()
				<< "\t" << (integrand_bnonE[NX *i + i].real() - integrand_bE[NX *i + i].real()) / integrand_bnonE[NX *i + i].real() << "\n";
		}

	}
	delete k;
}


void test_reduction();

void Load_trV(std::complex<double>* trace_V, const double number_rapidity, const int number) {


	std::ostringstream ifilename;
	ifilename << "G:\\hagiyoshi\\Data\\JIMWLK\\test_square\\re_v_over_Nc_expu_initial_unit_outside_sqare_proton_NX_"
		<< NX << "_INITN_" << INITIAL_N << "_num_" << number << "_" << number_rapidity << ".txt";

	//imput and output file
	std::ifstream ifs(ifilename.str().c_str());

	char str[256];
	if (ifs.fail())
	{
		std::cerr << "failed to load file " << number << "\t" << number_rapidity << std::endl;
	}

	ifs.getline(str, 256 - 1);


	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			ifs >> trace_V[NX*j + i];
			//std::cout << x[N*j + i] << "\t" << y[N*j + i] << "\t" << Solution[N*j + i]<< "\n";
		}
		//ofs_res_i << "\n";
		//std::cout << "\n";
	}

}

void JIMWLK_test(int position) {
	std::complex<double>* trace_V = new std::complex<double>[NX*NX];
	double* trace_V_square = new double[NX*NX];
	std::complex<double>* trace_V_temp = new std::complex<double>[NX*NX];
	std::vector<std::complex<double>> D_matrix(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;
	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = re*h;
	}

	for (int i = 0; i < NX*NX; i++) { trace_V[i] = 0; trace_V_square[i] = 0;}

	double number_rapidity = 0.01 / 3.0;
	for (int num = initial_number; num < number_of_comfig; num++) {
		Load_trV(trace_V_temp,number_rapidity,num);

		for (int i = 0; i < NX*NX; i++) { trace_V[i] += trace_V_temp[i];  
		trace_V_square[i] += trace_V_temp[i].real() * trace_V_temp[i].real();
		}
	}



	std::ostringstream ofilename_i2;
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\trV_inner_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< "_initial_" << initial_number << "_final_" << number_of_comfig << "_rapidity_" << number_rapidity << ".txt";

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t 1-V_matrix \t error \n";

	// re________
	//  |  |     |
	//  |  |__   |
	//  |        |
	//  |________|
	//     position
	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << setprecision(10) << relative_distance[re] << "\t" << trace_V[NX*(NX/2-position)+re+NX/2].real()/(double(number_of_comfig - initial_number))
			<< "\t" << sqrt( -trace_V[NX*(NX / 2 - position) + re + NX / 2].real() / (double(number_of_comfig - initial_number))
				*trace_V[NX*(NX / 2 - position) + re + NX / 2].real() / (double(number_of_comfig - initial_number))
				+ trace_V_square[NX*(NX / 2 - position) + re + NX / 2] / (double(number_of_comfig - initial_number)) )
			/sqrt((double(number_of_comfig - initial_number)) -1.0) << "\n";
	}

	delete[]trace_V;
	delete[]trace_V_square;
	delete[]trace_V_temp;
}

void JIMWLK_test_analytical(int position) {

	std::vector<std::complex<double>> D_matrix(NX / 2, 0), D_matrix_integ(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;
	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = re*h;
	}
	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	double number_rapidity = 0.01 / 3.0;
	double coeff = (double(Nc*Nc - 1.0)) / 2.0 / (double(Nc))*ALPHA_S / M_PI / M_PI*number_rapidity*mass*mass;

#pragma omp parallel for num_threads(6)
	for (int re = 0; re < NX / 2; re++) {
		if (double(re) <= double(NX*1.0 / 6.0)) {
			D_matrix[re] = std::complex<double>(-1.0/Nc,0.0);
		}
		else {

			for (int ix = 0; ix < NX; ix++) {
				for (int iy = 0; iy < NX; iy++) {
					//if(double(re + NX / 2.0-ix) > double(NX / 3.0) && double(re + NX / 2.0-ix)<double(2.0*NX / 3.0) 
					//	&& double( NX / 2.0 - position-iy) > double(NX / 3.0)  && double( NX / 2.0 - position-iy) < double(2.0*NX / 3.0) ){
					//	D_matrix[re] += std::complex<double>(modified_bessel1(mass*sqrt(x[NX*iy + ix] * x[NX*iy + ix] + y[NX*iy + ix] * y[NX*iy + ix]))
					//		*modified_bessel1(mass*sqrt(x[NX*iy + ix] * x[NX*iy + ix] + y[NX*iy + ix] * y[NX*iy + ix])),0.0);
					//}
					//else {

					//	D_matrix[re] +=0.0;
					//}
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					if (iy == 0 || iy == NX - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (iy % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {
						simpson1 = 4.0 / 3.0;
					}

					if (ix == 0 || ix == NX - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (ix % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {
						simpson2 = 4.0 / 3.0;
					}
					D_matrix[re] += std::complex<double>(simpson1*simpson2
						*modified_bessel1(mass
									*sqrt((-double(position)*h - x[NX*iy + ix]/3.0) * (-double(position)*h - x[NX*iy + ix]/3.0)
												+ (double(re)*h - y[NX*iy + ix]/3.0) * (double(re)*h - y[NX*iy + ix]/3.0) ))

						*modified_bessel1(mass
							*sqrt((-double(position)*h - x[NX*iy + ix] / 3.0) * (-double(position)*h - x[NX*iy + ix] / 3.0)
								+ (double(re)*h - y[NX*iy + ix] / 3.0) * (double(re)*h - y[NX*iy + ix] / 3.0)))
						, 0.0);

				}
			}

			D_matrix[re] = 1.0 - coeff*h/3.0*h/3.0* D_matrix[re];
			D_matrix_integ[re]= coeff*h/3.0*h/3.0* D_matrix[re];

		}

	}


	std::ostringstream ofilename_i2;
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\trV_inner_analytic_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		 << ".txt";

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t 1-V_matrix \t trV \n";

	// re________
	//  |  |     |
	//  |  |__   |
	//  |        |
	//  |________|
	//     position
	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 <<setprecision(10) << relative_distance[re] << "\t" << D_matrix[re].real() << "\t" << D_matrix_integ[re].real() << "\n";
	}

}


void JIMWLK_test_analytical_mini(int position) {

	std::vector<std::complex<double>> D_matrix(NX / 2, 0), D_matrix_integ(NX / 2, 0);

	double h = 1.0*LATTICE_SIZE / NX;
	std::vector<double> relative_distance(NX / 2, 0);
	for (int re = 0; re < NX / 2; ++re) {
		relative_distance[re] = re*h;
	}
	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	double number_rapidity = 0.01 / 3.0;
	double coeff = (double(Nc*Nc - 1.0)) / 2.0 / (double(Nc))*ALPHA_S / M_PI / M_PI*number_rapidity*mass*mass;

#pragma omp parallel for num_threads(6)
	for (int re = 0; re < NX / 2; re++) {
		if (double(re) <= double(NX*1.0 / 6.0)) {
			D_matrix[re] = std::complex<double>(-1.0 / Nc, 0.0);
		}
		else {

			for (int ix = NX / 3+1; ix < 2*NX / 3; ix++) {
				for (int iy = NX / 3+1; iy < 2*NX / 3; iy++) {
					//if(double(re + NX / 2.0-ix) > double(NX / 3.0) && double(re + NX / 2.0-ix)<double(2.0*NX / 3.0) 
					//	&& double( NX / 2.0 - position-iy) > double(NX / 3.0)  && double( NX / 2.0 - position-iy) < double(2.0*NX / 3.0) ){
					//	D_matrix[re] += std::complex<double>(modified_bessel1(mass*sqrt(x[NX*iy + ix] * x[NX*iy + ix] + y[NX*iy + ix] * y[NX*iy + ix]))
					//		*modified_bessel1(mass*sqrt(x[NX*iy + ix] * x[NX*iy + ix] + y[NX*iy + ix] * y[NX*iy + ix])),0.0);
					//}
					//else {

					//	D_matrix[re] +=0.0;
					//}
					double simpson1 = 1.0;
					double simpson2 = 1.0;
					if (iy == NX / 3 + 1 || iy == 2*NX / 3 - 1) {
						simpson1 = 1.0 / 3.0;
					}
					else if (iy % 2 == 0) {
						simpson1 = 2.0 / 3.0;
					}
					else {
						simpson1 = 4.0 / 3.0;
					}

					if (ix == NX / 3 + 1 || ix == 2*NX / 3 - 1) {
						simpson2 = 1.0 / 3.0;
					}
					else if (ix % 2 == 0) {
						simpson2 = 2.0 / 3.0;
					}
					else {
						simpson2 = 4.0 / 3.0;
					}
					D_matrix[re] += std::complex<double>(simpson1*simpson2
						*modified_bessel1(mass
							*sqrt((-double(position)*h - x[NX*iy + ix] ) * (-double(position)*h - x[NX*iy + ix])
								+ (double(re)*h - y[NX*iy + ix] ) * (double(re)*h - y[NX*iy + ix] )))

						*modified_bessel1(mass
							*sqrt((-double(position)*h - x[NX*iy + ix] ) * (-double(position)*h - x[NX*iy + ix] )
								+ (double(re)*h - y[NX*iy + ix] ) * (double(re)*h - y[NX*iy + ix] )))
						, 0.0);

				}
			}

			D_matrix[re] = 1.0 - coeff*h *h * D_matrix[re];
			D_matrix_integ[re] = coeff*h *h * D_matrix[re];

		}

	}


	std::ostringstream ofilename_i2;
	ofilename_i2 << "G:\\hagiyoshi\\Data\\JIMWLK\\output\\trV_inner_analytic_mini_position_RP_" << position << "_NX_" << NX << "_Nini_" << INITIAL_N
		<< ".txt";

	std::ofstream ofs_res_i2(ofilename_i2.str().c_str());

	ofs_res_i2 << "#relative distance \t 1-V_matrix \t trV \n";

	// re________
	//  |  |     |
	//  |  |__   |
	//  |        |
	//  |________|
	//     position
	for (int re = 1; re < NX / 2; ++re) {
		ofs_res_i2 << setprecision(10) << relative_distance[re] << "\t" << D_matrix[re].real() << "\t" << D_matrix_integ[re].real() << "\n";
	}

}


int main()
{


	std::complex<double>* V_initial = new std::complex<double>[3 * 3 * NX*NX];
	std::vector<std::complex<double>> D_matrix(NX/2,0);

	std::vector<double> x(NX*NX, 0), y(NX*NX, 0);
	double h = 1.0*LATTICE_SIZE / NX;
	double   xmax = h *NX / 2.0, xmin = -h*NX / 2.0, ymin = -h*NX / 2.0;
	for (int j = 0; j < NX; j++) {
		for (int i = 0; i < NX; i++)
		{
			x[NX*j + i] = xmin + i*h;
			y[NX*j + i] = ymin + j*h;
		}
	}

	Generator_SU3_initializer();
	//assemble_initial_Quark_position(number_of_comfig);
	//position * LATTICE_IZE/NX = impact_parameter
	int position = 32;
	int maxrap = 0;
	//Calculate_D_matrix(position,maxrap);
	//Calculate_onepoint_matrix(position, maxrap);
	//Integration_Smatrix(V_initial, maxrap);
	//Integration_Smatrix_towards_Wigner(V_initial, maxrap);
	//JIMWLK_test(position);

	//JIMWLK_test_analytical(position);
	//JIMWLK_test_analytical_mini(position);
	//MV_model_calculation_of_T_matrix();
	//MV_model_calculation_of_V_matrix(position);

	//calculate_c_2_4(maxrap);

	int maxmom = 40;
	//Derive_Wigner_distribution_revised(V_initial, maxrap,maxmom);
	//Derive_Wigner_distribution_fromSmatrix(V_initial, maxrap, maxmom);
	//Derive_Wigner_distribution_DP_WW(V_initial, maxrap, maxmom);
	//Derive_Wigner_distribution_DP_WW_diagonal(V_initial, maxrap, maxmom);
	//Wigner_DP_WW_diagonal(V_initial, maxrap, maxmom);
	//Derive_TMD_DP(V_initial, maxrap);
	//Derive_TMD_DP_direct(V_initial, maxrap);
	//Derive_GPD_direct(V_initial, maxrap);
	//MV_Wigner(maxmom);

	//test_reduction();

	delete[]V_initial;
}