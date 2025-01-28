# include "Layer.hpp"

Layer::Layer() {
	this->size = 0;
	this->type = 0;
}

Layer::Layer(int size, int type) {
	this->size = size;
	this->type = type;
	
	for (int i = 0; i < size; i++) {
		Neuron *n = new Neuron();
		neurons.emplace_back(n);
	}
	
	cout << "Layer created" << endl;
}

Layer::Layer(int size) {
	this->size = size;
	this->type = 0;
	
	for (int i = 0; i < size; i++) {
		Neuron *n = new Neuron();
		neurons.emplace_back(n);
	}
	
	cout << "Layer created" << endl;
}

Layer::~Layer() {
	for (int i = 0; i < size; i++) {
		delete neurons[i];
	}
	cout << "Layer destroyed" << endl;	
}

void	Layer::setValue(int i, double v) {
	neurons[i]->setValue(v);
}

Matrix *Layer::matrixifyValues() {
	Matrix *m = new Matrix(1, this->size, false);
	
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, neurons[i]->getValue());
	}
	
	return m;
}

Matrix *Layer::matrixifyActivatedValues() {
	Matrix *m = new Matrix(1, this->size, false);
	
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, neurons[i]->getActivatedValue());
	}
	
	return m;
}

Matrix *Layer::matrixifyDerivedValues() {
	Matrix *m = new Matrix(1, this->size, false);
	
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, neurons[i]->getDerivedValue());
	}
	
	return m;
}

void	Layer::print() {
	for (int i = 0; i < this->size; i++) {
		cout << "Neuron " << i << ": ";
		neurons[i]->print();
	}
}