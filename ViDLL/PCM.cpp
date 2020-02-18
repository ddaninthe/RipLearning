#include "PCM.h"
#include "pch.h"

double * createPCMModel(double* nbInputs) {
	auto t = new double[nbInputs[0]];
	return t;
}

void trainPCMClassification() {

}

void trainPCMRegression() {

}

double predictPCMClassification() {
	return rand() % 20 - 10;
}

double predictPCMRegression() {
	return rand() % 20 - 10;
}