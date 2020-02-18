#pragma once

#include "LinearModel.h"
#include "pch.h"

using namespace std;

double* createLinearModel(int nbInputs) {
	auto model = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs + 1; i++) {
		model[i] = ((rand() / (double) RAND_MAX) - 0.5) * 2;
	}
	return model;
}

/**
 * @param dataset the array containing data
 * @param dataSize the size of the data
 * @param model the linear model created, with the bias
 * @param modelSize the size of the model, minus the bias/predict
 * @param nbIter the number of iteration
 */
void trainLinearClassification(double* dataset, int dataSize, double* model, int modelSize, double nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		int g = predictLinearClassification(model, modelSize, data);
		double modif = learning * (data[modelSize] - g);

		model[modelSize] += modif;
		for (int k = 0; k < modelSize; k++) {
			model[k] += modif * data[k];
		}
	}
}


/**
 * @param dataset the array containing data
 * @param dataSize the size of the data
 * @param model the linear model created, with the bias
 * @param modelSize the size of the model, minus the bias/predict
 * @param nbIter the number of iteration
 */
void trainLinearRegression(double* dataset, int dataSize, double* model, int modelSize, double nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int random = rand() % dataSize;
		double *data = dataset + random * (modelSize + 1);

		double g = predictLinearRegression(model, modelSize, data);
		double modif = learning * (data[modelSize] - g);
		
		model[modelSize] += modif;
		for (int k = 0; k < modelSize; k++) {
			model[k] += modif * data[k];
		}
	}
}


int predictLinearClassification(double* model, int size, double* inputs) {
	return predictLinearRegression(model, size, inputs) >= 0 ? 1 : -1;
}

double predictLinearRegression(double* model, int size, double* inputs) {
	double res = model[size];
	for (int i = 0; i < size; i++) {
		res += model[i] * inputs[i];
	}
	return res;
}

void clear(double* ptr) {
	delete[] ptr;
}