#pragma once

extern "C" __declspec(dllexport) double* createPCMModel(int* layout);
extern "C" __declspec(dllexport) void trainPCMClassification();
extern "C" __declspec(dllexport) void trainPCMRegression();
extern "C" __declspec(dllexport) double predictPCMClassification();
extern "C" __declspec(dllexport) double predictPCMRegression();