#pragma once

extern "C" {
	typedef struct PCM {
		double** delta;
		double** x;
		double*** w;
		int* d;
		int size;
	} PCM;

	__declspec(dllexport) PCM* createPCMModel(int* layout, int arraySize);
	__declspec(dllexport) void trainPCMClassification(PCM* model, double* dataset, double* expect, int dataSize, int nbIter, double learning);
	__declspec(dllexport) void trainPCMRegression(PCM* model, double* dataset, double* expect, int dataSize, int nbIter, double learning);
	__declspec(dllexport) double* predictPCMClassification(PCM * model, double* data);
	__declspec(dllexport) double* predictPCMRegression(PCM * model, double* data);
}

double*** fillW(int* layout, int l);
double** fillArrayZero(int* layout, int l);
double** fillX(int* layout, int l);