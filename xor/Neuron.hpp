#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <math.h>

using namespace std;

class Neuron {
	public:
	double value;
	double activatedValue;
	double derivedValue;
	Neuron();
	Neuron(double value);
	~Neuron();
	void	setValue(double value);
	void	print();
	
	void	activate();
	void	derivate();
	
	double 		getValue();
	double		getActivatedValue();
	double		getDerivedValue();
};

#endif