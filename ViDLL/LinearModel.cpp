#pragma once
#include "LinearModel.h"
#include "pch.h"

int createLinearModel(int nbInputs, int nbOutputs) {
	return rand() % 20 - 10;
}

void trainLinearModel() {

}

double predictLinear(int model, int x, int y) {
	return rand() % 20 - 10;
}