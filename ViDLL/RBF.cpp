#include "RBF.h"
#include "pch.h"

using namespace Eigen;

RBF* createRBFModel(double* dataset, int datasetSize, int dimensions) {
	RBF* model = new RBF();
	model->coordinates = new double*[datasetSize];
	for (int i = 0; i < datasetSize; i++) {
		double* coor = new double[dimensions];
		for (int c = 0; c < dimensions; c++) {
			coor[c] = dataset[dimensions * i + c];
		}
		model->coordinates[i] = coor;
	}
	
	model->weights = new double[datasetSize];

	return model;
}

void trainNaiveRBF(RBF* model, int datasetSize, double* expectedOutputs, int dimensions, double gamma) {
	MatrixXd phi(datasetSize, datasetSize);
	VectorXd Y(datasetSize);

	for (int i = 0; i < datasetSize; i++) {
		for (int j = 0; j < datasetSize; j++) {
			phi(i, j) = exp(RBFExponent(gamma, model->coordinates[i], model->coordinates[j], dimensions));
		}
		
		Y(i) = expectedOutputs[i];
	}

	VectorXd W = phi.inverse() * Y;
	for (int i = 0; i < W.rows(); i++) {
		model->weights[i] = W(i);
	}
}

void trainRBFLloyd(RBF* model, int clusterCount, int datasetSize, double* expectedOutputs, int dimensions, double gamma) {
	// Cluster assignation
	double* dataPerCluster = new double[clusterCount];
	double** clusterData = new double*[clusterCount];
	double maxPerCluster = datasetSize / (double)clusterCount;
	for (int i = 0; i < clusterCount; i++) {
		dataPerCluster[i] = 0;
	}

	// Assign to cluster
	int data = 0, assigned = 0;
	while (assigned < datasetSize) {
		int cluster = rand() % clusterCount;
		if (dataPerCluster[cluster] < maxPerCluster * dimensions) {
			for (int dim = 0; dim < dimensions; dim++) {
				clusterData[cluster] = model->coordinates[assigned++];
			}
			dataPerCluster[cluster]++;
		}
	}

	// First representents
	double** representents = new double*[clusterCount];
	for (int i = 0; i < clusterCount; i++) {
		double* mu = new double[dimensions];
		for (int dim = 0; dim < maxPerCluster; dim++) {

		}
	}

	bool changed = true; // False when representants have not changed
	while (changed) {

	}

	// Train
	MatrixXd phi(datasetSize, clusterCount);
	VectorXd Y(datasetSize);
	for (int i = 0; i < datasetSize; i++) {
		for (int j = 0; j < clusterCount; j++) {
			phi(i, j) = exp(RBFExponent(gamma, model->coordinates[i], representents[j], dimensions));
		}
		Y(i) = expectedOutputs[i];
	}

	VectorXd W = ((phi.transpose() * phi).inverse() * phi.transpose()) * Y;
	for (int i = 0; i < W.rows(); i++) {
		model->weights[i] = W(i);
	}
}

int predictRBFClassification(RBF* model, double gamma, double* inputs, int dimensions, int modelSize) {
	return predictRBFRegression(model, gamma, inputs, dimensions, modelSize) >= 0 ? 1 : -1;
}

double predictRBFRegression(RBF* model, double gamma, double* inputs, int dimensions, int modelSize) {
	double res = 0;
	for (int i = 0; i < modelSize; i++) {
		double exponent = RBFExponent(gamma, inputs, model->coordinates[i], dimensions);
		res += model->weights[i] * exp(exponent);
	}
	return res;
}

double RBFExponent(double gamma, double* inputs, double* X, int dim) {
	return -gamma * squareMagnitude(inputs, X, dim);
}

double squareMagnitude(double* a, double* b, int dim) {
	double magn = 0;
	for (int i = 0; i < dim; i++) {
		magn += pow(a[i] - b[i], 2);
	}
	return magn;
}