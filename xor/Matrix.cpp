// # include "Layer.hpp"
# include "Matrix.hpp"
#include <random>

Matrix::Matrix() {
	this->rows = 0;
	this->cols = 0;
}

int	Matrix::getCols() {
	return this->cols;
}

int	Matrix::getRows() {
	return this->rows;
}

Matrix::Matrix(int rows, int cols, bool isRandom) {
	this->rows = rows;
	this->cols = cols;
	
	for (int i = 0; i < rows; i++) {
		vector<double> row;
		for (int j = 0; j < cols; j++) {
			if (isRandom) {
				row.emplace_back(this->generateRandomValue());
			} else {
				row.emplace_back(0.00);
			}
		}
		this->matrix.emplace_back(row);
	}
}

void	Matrix::print() {
	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			cout << this->getValue(i, j) << "\t\t\t";
		}
		cout << endl;
	}
}

double	Matrix::generateRandomValue() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	
	return dis(gen);
}

Matrix *Matrix::transpose() {
	Matrix *m = new Matrix(this->cols, this->rows, false);
	
	for (int i = 0; i < this->rows; i++) {
		for (int j = 0; j < this->cols; j++) {
			m->setValue(j, i, this->getValue(i, j));
		}
	}
	
	return m;
}

void	Matrix::setValue(int r, int c, double v) {
	matrix[r][c] = v;
}

double	Matrix::getValue(int r, int c) {
	return matrix[r][c];
}

Matrix::~Matrix() {
	cout << "Matrix destroyed" << endl;
}