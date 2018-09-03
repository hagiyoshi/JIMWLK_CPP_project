//one should use row = column.


#include <cstdio>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

/*
*This class aims to interpolate 2-dimentional function. 
*/
class interpolation_2dim
{
private:
	vector<double> M_func;
	vector<double> vec_x, vec_y;
	double x_max,x_min, y_max,y_min;
	int size_vector;
public:
	void set_points(const std::vector<double>& A,
		const std::vector<double>& y1, const std::vector<double>& y2);
	double operator() (double x,double y) const;
};


void interpolation_2dim::set_points(const std::vector<double>& A,
	const std::vector<double>& y1, const std::vector<double>& y2)
{
	M_func = A;
	vec_x = y1;
	vec_y = y2;
	size_vector = y1.size();

	x_max = vec_x[y1.size() -1];
	y_max = vec_y[y2.size() - 1];
	x_min = vec_x[0];
	y_min = vec_y[0];

}

double interpolation_2dim::operator() (double x, double y) const
{
	// find the closest point vec_x[idx] < x, idx=0 even if x<vec_x[0]
	std::vector<double>::const_iterator it;
	it = std::lower_bound(vec_x.begin(), vec_x.end(), x);
	int idx = std::max(int(it - vec_x.begin()) - 1, 0);

	// find the closest point vec_y[idy] < y, idy=0 even if y<vec_y[0]
	std::vector<double>::const_iterator ity;
	ity = std::lower_bound(vec_y.begin(), vec_y.end(), y);
	int idy = std::max(int(ity - vec_y.begin()) - 1, 0);

	int it_d = int(it - vec_x.begin());
	int ity_d = int(ity - vec_y.begin());

	//cout <<"y_max "<< y_max <<" x "<<x <<" y " <<y << " it "<< it_d << " idx " << idx << " ity " << ity_d << " idy " << idy<< endl;

	double t, u;

	if (idx+1 <= vec_x.size() - 1 && idy+1  <= vec_y.size() - 1){
		//bilinear interpolation
		 t = (x - vec_x[idx]) / (vec_x[idx + 1] - vec_x[idx]);
		 u = (y - vec_y[idy]) / (vec_y[idy + 1] - vec_y[idy]);
	}
	else{

		//bilinear interpolation
		t = (x - vec_x[vec_x.size() - 1 - 1]) / (vec_x[vec_x.size() - 1] - vec_x[vec_x.size() - 1 - 1]);
		u = (y - vec_y[vec_y.size() - 1 - 1]) / (vec_y[vec_y.size() - 1] - vec_y[vec_y.size() - 1 - 1]);
	}

	double interpole;

	//we set the value in the out of range as 0.
	if (x > x_max || y > y_max )
	{
		//std::cout << "out of range" << std::endl;
		//std::cout << "x " << x << " y " << y << std::endl;
		interpole = M_func[size_vector*size_vector - 1];
		//assert(1);
	}else if(x < x_min || y < y_min){

		interpole = M_func[0];
	}
	else
	{
		//f(x,y) = (1-t)(1-u)f(x_i,y_k)+t(1-u)f(x_i+1,y_k)+tuf(x_i+1,y_k+1)+(1-t)uf(x_i,y_k+1)
		interpole = (1.0 - t)*(1.0 - u)*M_func[idx + idy*size_vector] + t*(1.0 - u)*M_func[idx + 1 + idy*size_vector]
			+ t*u*M_func[idx + 1 + (idy + 1)*size_vector] + (1.0 - t)*u*M_func[idx + (idy + 1)*size_vector];
	}

	return interpole;
}