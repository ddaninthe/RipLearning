#pragma once

#include "LinearModel.h"
#include "pch.h"

using namespace Eigen;

LinearModel* createLinearModel(int nbInputs) {
	LinearModel* model = new LinearModel();
	model->weights = new double[nbInputs + 1];
	for (int i = 0; i < nbInputs + 1; i++) {
		model->weights[i] = ((rand() / (double) RAND_MAX) - 0.5) * 2;
	}
	model->isLinear = false;
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
void trainLinearClassification(double* dataset, int datasetSize, double* expectedOutputs, LinearModel* model, int modelSize, double nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int index = rand() % datasetSize;
		double *data = dataset + index * modelSize;

		int g = predictLinearClassification(model, modelSize, data);
		double modif = learning * (expectedOutputs[index] - g);

		model->weights[modelSize] += modif;
		for (int k = 0; k < modelSize; k++) {
			model->weights[k] += modif * data[k];
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
void trainLinearRegression(double* dataset, int datasetSize, double* expectedOutputs, LinearModel* model, int modelSize) {
	// Detect same line
	if (datasetSize >= 2 && modelSize == 2) {
		bool linear = true;

		double* ab = solve(dataset[0], dataset[1], dataset[2], dataset[3]);
		double a = ab[0], b = ab[1];

		double x, y;
		for (int i = 2; i < datasetSize; i++) {
			x = dataset[2 * i];
			y = dataset[2 * i + 1];
			if (!equals(y, a * x + b)) {
				linear = false;
				break;
			}
		}

		if (linear) {
			model->a = a;
			model->b = b;
			model->isLinear = true;
			return;
		}
	}

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
		model->weights[i] = W(i, 0);
	}
}

int predictLinearClassification(LinearModel* model, int size, double* inputs) {
	return predictLinearRegression(model, size, inputs) >= 0 ? 1 : -1;
}

double predictLinearRegression(LinearModel* model, int size, double* inputs) {
	if (model->isLinear) {
		return model->a * inputs[0] + model->b;
	}
	else {
		double res = model->weights[size];
		for (int i = 0; i < size; i++) {
			res += model->weights[i] * inputs[i];
		}
		return res;
	}
}

void clear(void* ptr) {
	delete[] ptr;
}

double* solve(double x1, double y1, double x2, double y2) {
	double a = (y1 - y2) / (x1 - x2);
	double b = y1 - (a * x1);
	return new double[2]{ a, b };
}

// Check double are equals
bool equals(double a, double b) {
	return abs(a - b) < pow(10, -6);
}