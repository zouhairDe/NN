#include "Neuron.hpp"

Neuron::Neuron() {
	this->value = 0;
	this->activatedValue = 0;
	this->derivedValue = 0;
}

Neuron::Neuron(double val) {
	this->value = val;
	activate();
	this->activatedValue = getActivatedValue();
	derivate();
	this->derivedValue = getDerivedValue();
}

Neuron::~Neuron() {
	cout << "Neuron deleted" << endl;
}

void Neuron::setValue(double val) {
	this->value = val;
}

void Neuron::activate() {
	this->activatedValue = 1 / (1 + exp(-this->value));
}

void Neuron::derivate() {
	this->derivedValue = this->activatedValue * (1 - this->activatedValue);
}

double Neuron::getValue() {
	return this->value;
}

double Neuron::getActivatedValue() {
	return this->activatedValue;
}

double Neuron::getDerivedValue() {
	return this->derivedValue;
}

void Neuron::print() {
	cout << "Value: " << this->value << endl;
	cout << "Activated Value: " << this->activatedValue << endl;
	cout << "Derived Value: " << this->derivedValue << endl;
}