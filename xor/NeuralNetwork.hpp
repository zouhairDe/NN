#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream> // IWYU pragma: keep
#include <vector>
#include "Matrix.hpp" // IWYU pragma: keep
#include "Layer.hpp" // IWYU pragma: keep

using namespace std;

class NeuralNetwork {
	private:
		vector<int> topology;
		size_t		topologySize;
		vector<double> input;
		vector<Layer *>		layers;
		vector<Matrix *>	weightsMatrices;
	public:
		NeuralNetwork();
		NeuralNetwork(vector<double> input);
		NeuralNetwork(vector<int> topology);
		~NeuralNetwork();
		
		void				feedForward(Matrix *input);
		void				backPropagation(Matrix *target);
		void				print();
		void				setTopology(vector<int> t);
		void				setCurrnetInput(vector<double> input);
		
		vector<int>			getTopology();
		
			

};
#endif