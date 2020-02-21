#include "MLP.h"
#include "pch.h"


MLP* createMLPModel(int* layout, int layoutSize) {
	MLP* model = new MLP();

	model->layers = new int[layoutSize];
	for (int i = 0; i < layoutSize; i++) {
		model->layers[i] = layout[i];
	}
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

void trainMLPClassification(MLP* model, double* dataset, double* expectedOutputs, int datasetSize, int iterations, double alpha) {
	int lastLayerInd = model->layersCount - 1;
	int lastLayerSize = model->layers[lastLayerInd];

	for (int it = 0; it < iterations; it++) {
		int index = rand() % datasetSize;
		double* inputs = dataset + index * model->layers[0]; // Get training inputs
		double* outputs = expectedOutputs + index * lastLayerSize; // training inputs'expected outputs

		// Fill x of last layer
		double* prediction = predictMLPClassification(model, inputs);
		for (int i = 0; i < lastLayerSize; i++) {
			model->x[lastLayerInd][i + 1] = prediction[i];
		}
		delete[] prediction;

		// Deltas last layer
		for (int j = 1; j < lastLayerSize + 1; j++) {
			double x = model->x[lastLayerInd][j];
			double delta = (1 - pow(x, 2)) * (x - outputs[j - 1]);
			model->deltas[lastLayerInd][j] = delta;
		}

		// Penultimate to first layers
		for (int l = lastLayerInd; l > 0; l--) {
			for (int i = 0; i < model->layers[l - 1] + 1; i++) {
				double wd = 0;
				for (int j = 1; j < model->layers[l] + 1; j++) {
					wd += model->weights[l][i][j] * model->deltas[l][j];
				}
				model->deltas[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * wd;
			}
		}

		// Update weights
		for (int l = 1; l <= lastLayerInd; l++) {
			for (int i = 0; i < model->layers[l - 1] + 1; i++) {
				for (int j = 1; j < model->layers[l] + 1; j++) {
					model->weights[l][i][j] -= alpha * model->x[l - 1][i] * model->deltas[l][j];
				}
			}
		}
	}
}

void trainMLPRegression(MLP* model, double* dataset, double* expectedOutputs, int datasetSize, int iterations, double alpha) {
	int lastLayerInd = model->layersCount - 1;
	int lastLayerSize = model->layers[lastLayerInd];

	for (int it = 0; it < iterations; it++) {
		int index = rand() % datasetSize;
		double* inputs = dataset + index * model->layers[0]; // Get training inputs
		double* outputs = expectedOutputs + index * lastLayerSize; // training inputs'expected outputs

		// Fill x of last layer
		double* prediction = predictMLPClassification(model, inputs);
		for (int i = 0; i < lastLayerSize; i++) {
			model->x[lastLayerInd][i + 1] = prediction[i];
		}
		delete[] prediction;

		// Deltas last layer
		for (int j = 1; j < lastLayerSize + 1; j++) {
			double x = model->x[lastLayerInd][j];
			double delta = x - outputs[j - 1];
			model->deltas[lastLayerInd][j] = delta;
		}

		// Penultimate to first layers
		for (int l = lastLayerInd; l >= 0; l--) {
			for (int i = 0; i < model->layers[l - 1] + 1; i++) {	
				double wd = 0;
				for (int j = 1; j < model->layers[l] + 1; j++) {
					wd += model->weights[l][i][j] * model->deltas[l][j];
				}
				model->deltas[l - 1][i] = (1 - pow(model->x[l - 1][i], 2)) * wd;
			}
		}

		// Update weights
		for (int l = 1; l <= lastLayerInd; l++) {
			for (int i = 0; i < model->layers[l - 1] + 1; i++) {
				for (int j = 1; j < model->layers[l] + 1; j++) {
					model->weights[l][i][j] -= alpha * model->x[l - 1][i] * model->deltas[l][j];
				}
			}
		}
	}
}


// Tanh to all results of Regression prediction
double* predictMLPClassification(MLP* model, double* inputs) {
	double* res = predictMLPRegression(model, inputs);

	for (int i = 0; i < model->layers[model->layersCount - 1]; i++) {
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
		for (int j = 1; j < layerSize + 1; j++) {
			model->x[layer][j] = 0;
			for (int i = 0; i < previousLayerSize + 1; i++) { // +1 for bias
				model->x[layer][j] += model->x[layer - 1][i] * model->weights[layer][i][j];
			}
		}

		// Tanh of sigmas but the last layer
		if (layer < model->layersCount - 1) {
			for (int j = 1; j < layerSize + 1; j++) {
				model->x[layer][j] = tanh(model->x[layer][j]);
			}
		}
	}

	// Return last layer without bias
	double* results = new double[model->layers[model->layersCount - 1]];
	for (int i = 0; i < model->layers[model->layersCount - 1]; i++) {
		results[i] = model->x[model->layersCount - 1][i + 1];
	}

	return results;
}