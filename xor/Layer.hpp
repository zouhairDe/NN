#ifndef LAYER_HPP
#define LAYER_HPP

#include "Matrix.hpp"
#include "Neuron.hpp"
#include <vector>
using namespace std;

class Layer {
	private:
		int size;
		int type;
		vector<Neuron *> neurons;
	public:
		Layer();
		Layer(int size, int type);
		Layer(int size);
		~Layer();
		
		void	setValue(int i, double v);
		Matrix	*matrixifyValues();
		Matrix	*matrixifyActivatedValues();
		Matrix	*matrixifyDerivedValues();
		
		void	print();
		
};

#endif