#include <vector>
#include <complex>
#include <Eigen/Dense>


std::complex<double> Generator_adjt[8][3][3];

void Generator_SU3_initializer(){
	//if you want to use double you should change Matrix3cf (float) into Matrix3cd (double)
Eigen::Matrix3cd t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8;
t_1 << std::complex<double>(0.0, 0.0), std::complex<double>(1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

t_2 << std::complex<double>(0.0, 0.0), std::complex<double>(0.0, -1.0 / 2.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 1.0 / 2.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

t_3 << std::complex<double>(1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(-1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

t_4 << std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.0 / 2.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

t_5 << std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, -1.0 / 2.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 1.0 / 2.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0);

t_6 << std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.0 / 2.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(1.0 / 2.0, 0.0), std::complex<double>(0.0, 0.0);

t_7 << std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, -1.0 / 2.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 1.0 / 2.0), std::complex<double>(0.0, 0.0);

t_8 << std::complex<double>(1.0 / sqrt(3.0) / 2.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(1.0 / sqrt(3.0) / 2.0), std::complex<double>(0.0, 0.0),
std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(-2.0 / sqrt(3.0) / 2.0, 0.0);


	std::complex<double> t1[3][3] = {
		{ (0.0 , 0.0), (1.0 / 2.0, 0.0), (0.0, 0.0) },
		{ (1.0 / 2.0, 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t2[3][3] = {
		{ (0.0 , 0.0), (0.0, -1.0 / 2.0), (0.0, 0.0) },
		{ (0.0, 1.0 / 2.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t3[3][3] = {
		{ (1.0 / 2.0 , 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (-1.0 / 2.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t4[3][3] = {
		{ (0.0 , 0.0), (0.0, 0.0), (1.0 / 2.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (1.0 / 2.0, 0.0), (0.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t5[3][3] = {
		{ (0.0 , 0.0), (0.0, 0.0), (0.0, -1.0 / 2.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 1.0 / 2.0), (0.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t6[3][3] = {
		{ (0.0 , 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (1.0 / 2.0, 0.0) },
		{ (0.0, 0.0), (1.0 / 2.0, 0.0), (0.0, 0.0) }
	};

	std::complex<double> t7[3][3] = {
		{ (0.0 , 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (0.0, -1.0 / 2.0) },
		{ (0.0, 0.0), (0.0, 1.0 / 2.0), (0.0, 0.0) }
	};

	std::complex<double> t8[3][3] = {
		{ (1.0 / sqrt(3.0) / 2.0 , 0.0), (0.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (1.0 / sqrt(3.0) / 2.0, 0.0), (0.0, 0.0) },
		{ (0.0, 0.0), (0.0, 0.0), (-2.0 / sqrt(3.0) / 2.0, 0.0) }
	};



	for (int a = 0; a < ADJNc; ++a) {
		if (a == 0) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_1(i,j);
				}
			}
		}
		else if (a == 1) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_2(i, j);
				}
			}
		}
		else if (a == 2) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_3(i, j);
				}
			}
		}
		else if (a == 3) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_4(i, j);
				}
			}
		}
		else if (a == 4) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_5(i, j);
				}
			}
		}
		else if (a == 5) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_6(i, j);
				}
			}
		}
		else if (a == 6) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_7(i, j);
				}
			}
		}
		else if (a == 7) {

			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {

					Generator_adjt[a][i][j] = t_8(i, j);
				}
			}
		}

	}
}