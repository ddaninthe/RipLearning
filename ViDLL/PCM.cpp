#include "PCM.h"
#include "pch.h"

PCM* createPCMModel(int* layout, int arraySize) {
	PCM* model = new PCM();
	model->d = layout;
	model->w = fillW(layout, arraySize);
	model->delta = fillArrayZero(layout, arraySize);
	model->x = fillX(layout, arraySize);
	model->size = arraySize;
	return model;
}

double*** fillW(int* layout, int l) {
	double*** w = new double**[l];
	int i, j;
	for (int a = 1; a < l; a++) {
		i = layout[a - 1], j = layout[a];
		w[a] = new double*[i];
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
	double** array = new double*[l];
	int j;
	for (int a = 0; a < l; a++) {
		j = layout[a];
		array[a] = new double[j];
	}
	return array;
}

double** fillX(int* layout, int l) {
	double** array = new double*[l];
	int j;
	for (int a = 0; a < l; a++) {
		j = layout[a];
		array[a] = new double[j + 1];
		array[a][0] = 1.0;
	}
	return array;
}

void trainPCMClassification(PCM* model, double* dataset, double* expect, int dataSize, int nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int random = rand() % dataSize;
		double* data = dataset + random * (model->d[0]);

		//prediction
		predictPCMClassification(model, data);
		
		int l = model->size;
		//delta initial
		for (int j = 1; j < model->d[l]; j++) {
			model->delta[l][j] = (1 - pow(model->x[l][j], 2)) * (model->x[l][j] - expect[random * model->d[model->size - 1] + j]);
		}

		//delta intermediaire
		for (int l = model->size; l > 0; l--) {
			for (int i = 1; i < model->d[l-1]; i++) {
				double sum = 0;
				for (int j = 1; j < model->d[l]; j++){
					sum += model->w[l][i][j] * model->delta[l][j];
				}
				model->delta[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * sum;
			}
		}

		//update w
		for (int l = 1; l < model->size; l++) {
			for (int j = 1; j < model->d[l]; j++) {
				for (int i = 0; i < model->d[l-1]; i++) {
					model->w[l][i][j] -= (learning * model->x[l - 1][i] * model->delta[l][j]);
				}
			}
		}
	}
}

void trainPCMRegression(PCM* model, double* dataset, double* expect, int dataSize, int nbIter, double learning) {
	for (int i = 0; i < nbIter; i++) {
		int random = rand() % dataSize;
		double* data = dataset + random * (model->d[0]);

		//prediction
		predictPCMRegression(model, data);

		int l = model->size;
		//delta initial
		for (int j = 1; j < model->d[l]; j++) {
			model->delta[l][j] = (1 - pow(model->x[l][j], 2)) * (model->x[l][j] - expect[random * model->d[model->size - 1] + j]);
		}

		//delta intermediaire
		for (int l = model->size; l > 0; l--) {
			for (int i = 1; i < model->d[l - 1]; i++) {
				double sum = 0;
				for (int j = 1; j < model->d[l]; j++) {
					sum += model->w[l][i][j] * model->delta[l][j];
				}
				model->delta[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * sum;
			}
		}

		//update w
		for (int l = 1; l < model->size; l++) {
			for (int j = 1; j < model->d[l]; j++) {
				for (int i = 0; i < model->d[l - 1]; i++) {
					model->w[l][i][j] -= (learning * model->x[l - 1][i] * model->delta[l][j]);
				}
			}
		}
	}
}

double* predictPCMClassification(PCM* model, double* data) {
	// add input in l=0
	for (int j = 1; j < model->d[0] + 1; j++) {
		model->x[0][j] = data[j-1];
	}
	double sum = 0;
	for (int l = 1; l < model->size; l++) {
		for (int j = 1; j < model->d[l] + 1; j++) {
			sum = 0.0;
			for (int i = 0; i < model->d[l]; i++) {
				sum += model->x[l - 1][i] * model->w[l][i][j];
			}
			model->x[l][j] = tanh(sum);
		}	
	}

	// ne pas retourner le premier !
	double* results = new double[model->size - 1];
	for (int i = 0; i < model->size - 1; i++) {
		results[i] = model->x[model->size - 1][i + 1];
	}
	return results;
}

double* predictPCMRegression(PCM* model, double* data) {
	// add input in l=0
	for (int j = 1; j < model->d[0] + 1; j++) {
		model->x[0][j] = data[j-1];
	}

	for (int l = 1; l < model->size - 1; l++) {
		for (int j = 1; j < model->d[l] + 1; j++) {
			double sum = 0.0;
			for (int i = 0; i < model->d[l]; i++) {
				sum += model->x[l - 1][i] * model->w[l][i][j];
			}
			model->x[l][j] = tanh(sum);
		}
	}
	for (int j = 1; j < model->d[model->size - 1] + 1; j++) {
		double sum = 0.0;
		for (int i = 0; i < model->d[model->size - 1]; i++) {
			sum += model->x[model->size - 1][i] * model->w[model->size - 1][i][j];
		}
		model->x[model->size - 1][j] = sum;
	}

	return model->x[model->size - 1];
}