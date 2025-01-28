# include "Layer.hpp" // IWYU pragma: keep
# include "Matrix.hpp" // IWYU pragma: keep
# include "NeuralNetwork.hpp" // IWYU pragma: keep
#include <vector>

int main()
{
	vector<int> config = {3, 2, 3};
	
	vector<double> input = {1, 0, 1};
	
	NeuralNetwork *nn = new NeuralNetwork(config);
	nn->setCurrnetInput(input);
	
	nn->print();
	return 0;
}