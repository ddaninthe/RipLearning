#pragma once

extern "C" __declspec(dllexport) double* createPCMModel(double * nbInputs);
extern "C" __declspec(dllexport) double* trainPCMClassification();
extern "C" __declspec(dllexport) double* trainPCMRegression();
extern "C" __declspec(dllexport) double predictPCMClassification();
extern "C" __declspec(dllexport) double predictPCMRegression();