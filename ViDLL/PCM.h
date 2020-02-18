#pragma once

extern "C" {
	typedef struct MLP{
		double** deltas;
		double** x;
		double*** w;
		int* d;
		int size;
	} MLP;

	__declspec(dllexport) MLP * createPCMModel(int* layout);
	__declspec(dllexport) void trainPCMClassification();
	__declspec(dllexport) void trainPCMRegression();
	__declspec(dllexport) double* predictPCMClassification(MLP * model, double* data);
	__declspec(dllexport) double* predictPCMRegression(MLP * model, double* data);
}