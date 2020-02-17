#pragma once
#include "LinearModel.h"
#include "pch.h"


double* createModel(int nbInputs) {
	auto model = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs; i++) {
		int rnd = rand() % 2;
		model[0] = rnd ? -1 : 1;
	}
	return model;
}

double* trainLinearModel(double dataset[], int dataSize, double* model, int modelSize, double iterNumber, double learning) {
	int random = 0;
	for (int i = 0; i < iterNumber; i++)
	{
		random = rand() % dataSize;
		double *data = dataset + random * modelSize;

		double g = predictLinear(model, modelSize-1, data);

		double modif = learning * (data[modelSize-1] - g);

		for (int k = 0; k < modelSize; k++)
		{
			model[k] += modif * model[k];
		}
	}

	return model;
	
}

double predictLinear(double* model, int size, double *inputs) {
	double res = 0;
	for (int i = 0; i < size; i++) {
		res += model[i] * inputs[i];
	}
	return res;
}

void clear(double* ptr) {
	delete[] ptr;
}