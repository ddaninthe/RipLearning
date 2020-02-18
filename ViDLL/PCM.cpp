#include "PCM.h"
#include "pch.h"

struct MLP {
	double** deltas;
	double** x;
	double*** w;
	int* d;
	int size;
};

double* createPCMModel(int* layout, int arraySize) {
	MLP model;
	model.d = layout;
	model.w = fillW(layout, arraySize);
	model.deltas = fillArrayZero(layout, arraySize);
	model.x = fillArrayZero(layout, arraySize);
	model.size = arraySize;
	return
}

double*** fillW(int* layout, int l) {
	double*** w = new double** [l];
	int i = 0, j = 0;
	for (int a = 1; a < l; a++) {
		i = layout[a - 1], j = layout[a];
		w[a] = new double* [i];
		for (int b = 0; b < i; b++) {
			w[a][b] = new double[j];
				for (int c = 0; c < j; c++) {
					w[a][b][c] = ((rand() / (double)RAND_MAX) - 0.5) * 2;
			}
		}
	}
	return w;
}

double** fillArrayZero(int* layout, int l) {
	double** array = new double* [l];
	int j = 0;
	for (int a = 1; a < l; a++) {
		j = layout[a];
		array[a] = new double[j];
	}
	return array;
}

void trainPCMClassification() {

}

void trainPCMRegression() {

}

double predictPCMClassification(MLP model) {
	
}

double predictPCMRegression() {
	return rand() % 20 - 10;
}