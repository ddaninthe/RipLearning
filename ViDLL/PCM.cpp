#include "PCM.h"
#include "pch.h"

double* createPCMModel(int* layout, int size) {
	double* model = new double[size];
	for (int i = 0; i < size; i++) {
		model[i] = createCellArray(layout[i])
	}
	return model;
}

double* createCellArray(int cellNumber) {
	double* cellArray = new double[cellNumber];
	for (int j = 0; j < cellNumber; j++) {
		cellArray[j] = ((rand() / (double)RAND_MAX) - 0.5) * 2;
	}
	return cellArray
}

void trainPCMClassification() {

}

void trainPCMRegression() {

}

double predictPCMClassification() {
	return rand() % 20 - 10;
}

double predictPCMRegression() {
	return rand() % 20 - 10;
}