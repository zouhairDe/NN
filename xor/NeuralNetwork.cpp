#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() { }

NeuralNetwork::~NeuralNetwork() {
	cout << "NeuralNetwork destroyed" << endl;
}

void	NeuralNetwork::setCurrnetInput(vector<double> input) {
	this->input = input;
	
	for (int i = 0; i < this->topology.size(); i++) {
		this->layers[0]->setValue(i, input[i]);
	}
}

NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topology = topology;//Config
	
	
	for (int i = 0; i < topology.size(); i++) {
		Layer *l = new Layer(topology[i]);
		this->layers.emplace_back(l);
	}
	
	for (int i = 0; i < topology.size() - 1; i++) {
		Matrix *m = new Matrix(topology[i], topology[i + 1], true);
		this->weightsMatrices.emplace_back(m);
	}
}

void	NeuralNetwork::print() {
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "Layer:	" << i << endl;
		if (i == 0) {
			Matrix *m = this->layers[i]->matrixifyValues();
			m->print();
		} else {
			Matrix *m = this->layers[i]->matrixifyActivatedValues();
			m->print();
		}
	}
	
	// for (int i = 0; i < this->weightsMatrices.size(); i++) {
	// 	this->weightsMatrices[i]->print();
	// 	cout << endl;
	// }
}