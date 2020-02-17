#pragma once
#include "LinearModel.h"
#include "pch.h"


double* createLinear(int nbInputs) {
	auto model = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs; i++) {
		model[0] = rand() % 3 - 1;
	}
	return model;
}

void trainLinearModel() {

}

double predictLinear(double* model, int size, double inputs[]) {
	double res = 0;
	for (int i = 0; i < size; i++) {
		res += model[i] * inputs[i];
	}
	return res;
}

void clear(double* ptr) {
	delete[] ptr;
}