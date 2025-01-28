#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream> // IWYU pragma: keep
#include <random> // IWYU pragma: keep

using namespace std;

class Matrix {
	private:
	int						rows;
	int						cols;
	// bool					isRandom;
	
	vector<vector<double>>	matrix;
	public:
	Matrix();
	Matrix(int rows, int cols, bool isRandom);
	~Matrix();
	Matrix	*transpose();
	void	setValue(int r, int c, double v);
	void	print();
	int		getRows();
	int		getCols();
	double	getValue(int r, int c);
	double 	generateRandomValue();
};


#endif