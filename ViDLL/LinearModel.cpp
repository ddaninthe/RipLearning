#pragma once

#include "LinearModel.h"
#include "pch.h"

using namespace Eigen;

double* createLinearModel(int nbInputs) {
	auto model = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs + 1; i++) {
		model[i] = ((rand() / (double) RAND_MAX) - 0.5) * 2;
	}
	return model;
}

/**
 * @param dataset the array containing data
 * @param datasetSize the size of the dataset
 * @param expectedOutputs the expectedOutputs
 * @param model the linear model created, with the bias
 * @param modelSize the size of the model, minus the bias/predict
 * @param nbIter the number of iteration
 */
void trainLinearClassification(double* dataset, int datasetSize, double* expectedOutputs, double* model, int modelSize, double nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int index = rand() % datasetSize;
		double *data = dataset + index * modelSize;

		int g = predictLinearClassification(model, modelSize, data);
		double modif = learning * (expectedOutputs[index] - g);

		model[modelSize] += modif;
		for (int k = 0; k < modelSize; k++) {
			model[k] += modif * data[k];
		}
	}
}

/**
 * @param dataset the array containing data
 * @param datasetSize the size of the dataset
 * @param expectedOutputs the value expected
 * @param model the linear model created, with the bias
 * @param modelSize the size of the model, minus the bias/predict
 * @param nbIter the number of iteration
 */
void trainLinearRegression(double* dataset, int datasetSize, double* expectedOutputs, double* model, int modelSize) {
	MatrixXd X(datasetSize, modelSize + 1);
	MatrixXd Y(datasetSize, 1);
	for (int i = 0; i < datasetSize; i++) {
		X(i, modelSize) = 1;
		for (int j = 0; j < modelSize; j++) {
			X(i, j) = dataset[2 * i + j];
		}
		Y(i, 0) = expectedOutputs[i];
	}

	MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;
	for (int i = 0; i < W.rows(); i++) {
		model[i] = W(i, 0);
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