#pragma once


extern "C" __declspec(dllexport) double* createLinearModel(int nbInputs);
extern "C" __declspec(dllexport) double* trainLinearModel(double dataset[], int ptr, double iterNumber);
extern "C" __declspec(dllexport) double predictLinear(double* ptr, int size, double inputs[]);
extern "C" __declspec(dllexport) void clear(double ptr);