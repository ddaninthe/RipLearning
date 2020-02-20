#pragma once
#include <pch.h>

extern "C" {
	typedef struct MLP{
		double** delta;
		double** x;
		double*** w;
		int* d;
		int size;
	} MLP;

	__declspec(dllexport) MLP* createPCMModel(int* layout, int arraySize);
	__declspec(dllexport) void trainPCMClassification(MLP* model, double* dataset, double* expect, int dataSize, int nbIter, double learning);
	__declspec(dllexport) void trainPCMRegression(MLP* model, double* dataset, double* expect, int dataSize, int nbIter, double learning);
	__declspec(dllexport) double* predictPCMClassification(MLP * model, double* data);
	__declspec(dllexport) double* predictPCMRegression(MLP * model, double* data);
}

double*** fillW(int* layout, int l);
double** fillArrayZero(int* layout, int l);
double** fillX(int* layout, int l);