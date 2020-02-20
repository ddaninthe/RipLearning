#include "MLP.h"
#include "pch.h"


MLP* createMLPModel(int* layout, int layoutSize) {
	MLP* model = new MLP();
	model->layers = layout;
	model->layersCount = layoutSize;

	model->deltas = new double*[layoutSize];
	model->x = new double*[layoutSize];

	// Deltas
	for (int lay = 0; lay < layoutSize; lay++) {
		model->deltas[lay] = new double[layout[lay] + 1];
		for (int i = 0; i < layout[lay] + 1; i++) {
			model->deltas[lay][i] = 0;
		}
	}

	// X
	for (int lay = 0; lay < layoutSize; lay++) {
		model->x[lay] = new double[layout[lay] + 1];
		for (int i = 0; i < layout[lay] + 1; i++) {
			model->x[lay][i] = i == 0 ? 1 : 0;  // Bias to 1
		}
	}

	// Weights
	model->weights = new double**[layoutSize];
	model->weights[0] = new double*[0];
	for (int lay = 1; lay < layoutSize; lay++) {
		int size = layout[lay] + 1; // +1 for bias
		model->weights[lay] = new double*[layout[lay - 1] + 1];

		for (int i = 0; i < layout[lay - 1] + 1; i++) {
			model->weights[lay][i] = new double[size];
			model->weights[lay][i][0] = 0;
			for (int j = 1; j < size; j++) {
				model->weights[lay][i][j] = (rand() / (double)RAND_MAX - 0.5) * 2;
			}
		}
	}

	return model;
}

double* predictMLPClassification(MLP* model, double* inputs) {
	double* res = predictMLPRegression(model, inputs);
	int size = model->layers[model->layersCount - 1];

	for (int i = 0; i < size; i++) {
		res[i] = tanh(res[i]);
	}
	return res;
}

double* predictMLPRegression(MLP* model, double* inputs) {
	// Set inputs
	for (int i = 0; i < model->layers[0]; i++) {
		model->x[0][i + 1] = inputs[i];
	}

	for (int layer = 1; layer < model->layersCount; layer++) {
		int layerSize = model->layers[layer];
		int previousLayerSize = model->layers[layer - 1];

		// Sigmas
		for (int i = 0; i < previousLayerSize + 1; i++) { // +1 for bias
			for (int j = 1; j < layerSize + 1; j++) {
				model->x[layer][j] += model->x[layer - 1][i] * model->weights[layer][i][j];
			}
		}

		// Tanh of sigmas but the last layer
		if (layer < model->layersCount - 1) {
			for (int i = 1; i < layerSize + 1; i++) {
				model->x[layer][i] = tanh(model->x[layer][i]);
			}
		}
	}

	double* results = new double[model->layers[model->layersCount - 1]];
	for (int i = 0; i < model->layers[model->layersCount - 1]; i++) {
		results[i] = model->x[model->layersCount - 1][i + 1];
	}

	return results;
}