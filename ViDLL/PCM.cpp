#include "PCM.h"
#include "pch.h"

double* createPCMModel(int* layout, int size, int cellSize) {
	double* model = new double[size];
	for (int i = 0; i < size; i++) {
		model[i] = createCellArray(layout[i], cellSize);
	}
	return model;
}

double* createCellArray(int cellNumber, int cellSize) {
	double* cellArray = new double[cellNumber];
	for (int j = 0; j < cellNumber; j++) {
		cellArray[j] = createCell(int cellSize);
	}
	return cellArray;
}

double* createCell(int cellSize) {
	double* cell = new double[cellSize];
	for (int k = 0; k < cellSize; j++) {
		cell[j] = ((rand() / (double)RAND_MAX) - 0.5) * 2;
	}
	return cell;
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