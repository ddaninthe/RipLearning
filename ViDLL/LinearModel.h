#pragma once

#ifdef LINEAR_EXPORTS
#define LINEAR_API __declspec(dllexport)
#else
#define LINEAR_API __declspec(dllimport)
#endif


extern "C" int createLinearModel(int nbInputs, int nbOutput);
extern "C" void trainLinearModel();
extern "C" double predictLinear(int model, int x, int y);