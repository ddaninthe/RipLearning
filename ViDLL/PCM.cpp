#include "PCM.h"
#include "pch.h"

MLP* createPCMModel(int* layout, int arraySize) {
	MLP* model = new MLP();
	model->d = layout;
	model->w = fillW(layout, arraySize);
	model->delta = fillArrayZero(layout, arraySize);
	model->x = fillX(layout, arraySize);
	model->size = arraySize;
	return model;
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

double** fillX(int* layout, int l) {
	double** array = new double* [l];
	int j = 0;
	for (int a = 1; a < l; a++) {
		j = layout[a];
		array[a] = new double[j+1];
		array[a][0] = 1.0;
	}
	return array;
}

void trainPCMClassification(MLP* model, double* dataset, double* predict, int dataSize, int nbIter) {

	for (int i = 0; i < nbIter; i++) {
		int random = rand() % dataSize;
		double* data = dataset + random * (model->size);

		//prediction
		predict = predictPCMClassification(model, data);
		
		int l = model->size;
		//delta initial
		for (int j = 1; j < model->d[l]; j++) {
			model->delta[l][j] = (1 - pow(model->x[l][j], 2)) * (model->x[l][j] - predict[j]);
		}

		//delta intermediaire
		for (int l = model->size; l > 0; l--) {
			for (int i = 1; i < model->d[l-1]; i++) {
				double sum = 0;
				for (int j = 1; j < model->d[l]; j++)
					sum += model->w[l][i][j] * model->delta[l][j];
			}
		}
	}
}

void trainPCMRegression() {

}

double* predictPCMClassification(MLP* model, double* data) {
	// add input in l=0
	for (int j = 1; j < model->d[0] + 1; j++) {
		model->x[0][j] = data[j];
	}
	double sum = 0;
	for (int l = 1; l < model->size; l++) {
		for (int j = 1; j < model->d[l] + 1; j++) {
			
			for (int i = 0; i < model->d[l]; i++) {
				sum += model->x[l - 1][i] * model->w[l][i][j];
			}
			model->x[l][j] = tanh(sum);
		}	
	}

	return model->x[model->size];
}

double* predictPCMRegression(MLP* model, double* data) {
	// add input in l=0
	for (int j = 1; j < model->d[0] + 1; j++) {
		model->x[0][j] = data[j];
	}

	for (int l = 1; l < model->size-1; l++) {
		for (int j = 1; j < model->d[l] + 1; j++) {
			double sum = 0.0;
			for (int i = 0; i < model->d[l]; i++) {
				sum += model->x[l - 1][i] * model->w[l][i][j];
			}
			model->x[l][j] = tanh(sum);
		}
	}
	for (int j = 1; j < model->d[model->size] + 1; j++) {
		double sum = 0.0;
		for (int i = 0; i < model->d[model->size]; i++) {
			sum += model->x[model->size - 1][i] * model->w[model->size][i][j];
		}
		model->x[model->size][j] = sum;
	}

	return model->x[model->size];
}