#pragma once


extern "C" __declspec(dllexport) double* createModel(int nbInputs);
extern "C" __declspec(dllexport) void trainLinearModel();
extern "C" __declspec(dllexport) double predictLinear(double* ptr, int size, double inputs[]);
extern "C" __declspec(dllexport) void clear(double* ptr);