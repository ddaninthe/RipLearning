#include "MLP.h"
#include "pch.h"


MLP* createMLPModel(int* layout, int layoutSize) {
	MLP* model = new MLP();
	model->layers = layout;
	model->layersCount = layoutSize;

	model->deltas = new double*[layoutSize];
	model->x = new double*[layoutSize];
	model->weights = new double**[layoutSize];

	for (int lay = 0; lay < layoutSize; lay++) {
		int size = layout[lay];
		model->deltas[lay] = new double[size + 1];
		model->x[lay] = new double[size + 1];
		model->weights[lay] = new double*[size + 1];

		for (int i = 0; i < size; i++) {
			model->deltas[lay][i] = 0;
			model->x[lay][i] = i == 0 ? 1 : 0;  // Bias to 1

			model->weights[lay + 1][i] = new double[layout[lay]];
			for (int j = 0; j < layout[lay + 1]; j ++) {
				model->weights[lay + 1][i][j] = (rand() / (double)RAND_MAX - 0.5) * 2;
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
	for (int layer = 1; layer < model->layersCount; layer++) {
		int layerSize = model->layers[layer];
		int previousLayerSize = model->layers[layer - 1];

		// Sigmas
		for (int i = 0; i < previousLayerSize + 1; i++) { // +1 for bias
			for (int j = 0; j < layerSize; j++) {
				model->x[layer][j] += model->x[layer - 1][i] * model->weights[layer][i][j];
			}
		}

		// Tanh of sigmas but the last
		if (layer < model->layersCount - 1) {
			for (int i = 0; i < layerSize; i++) {
				model->x[layer][i] = tanh(model->x[layer][i]);
			}
		}
	}

	return model->x[model->layers[model->layersCount - 1]];
}