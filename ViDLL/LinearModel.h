#pragma once

#ifdef LINEAR_EXPORTS
#define LINEAR_API __declspec(dllexport)
#else
#define LINEAR_API __declspec(dllimport)
#endif

extern "C" double* createLinearModel(int nbInputs);
extern "C" void trainLinearModel();
extern "C" double predictLinear(double* ptr, int size, double inputs[]);
extern "C" void clear(double ptr);