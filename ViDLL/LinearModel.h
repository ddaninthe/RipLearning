#pragma once

extern "C" __declspec(dllexport) double* createLinearModel(int nbInputs);
extern "C" __declspec(dllexport) double* trainLinearClassification(double* dataset, int dataSize, double* model, int modelSize, double iterNumber, double learning);
extern "C" __declspec(dllexport) double* trainLinearRegression(double* dataset, int dataSize, double* model, int modelSize, double iterNumber, double learning);
extern "C" __declspec(dllexport) double predictLinearClassification(double* ptr, int size, double* inputs);
extern "C" __declspec(dllexport) double predictLinearRegression(double* ptr, int size, double* inputs);
extern "C" __declspec(dllexport) void clear(double* ptr);