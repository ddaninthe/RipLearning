#pragma once

extern "C" {
	typedef struct MLP{
		double** delta;
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

double*** fillW(int* layout, int l);
double** fillArrayZero(int* layout, int l);
double** fillX(int* layout, int l);