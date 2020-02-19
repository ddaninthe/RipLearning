﻿using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class MLPSepLinear2D3EScript : MonoBehaviour
{
    /*
     MLP* createPCMModel(int* layout, int arraySize);
    void trainPCMClassification(MLP* model, double* dataset, double* predict, int dataSize, int nbIter, double learning);
    void trainPCMRegression(MLP* model, double* dataset, double* predict, int dataSize, int nbIter, double learning);
    double* predictPCMClassification(MLP * model, double* data);
    double* predictPCMRegression(MLP * model, double* data);
     */


    [DllImport("ViDLL.dll")]
    private static extern IntPtr createPCMModel(int[] layout, int arraySize);

    [DllImport("ViDLL.dll")]
    private static extern void trainPCMClassification(IntPtr model, double[] dataset, double[] expectedOutputs, int datasetSize, double nbIter, double learning);

    [DllImport("ViDLL.dll")]
    private static extern double[] predictPCMClassification(IntPtr model, double[] data);

    [DllImport("ViDLL.dll")]
    private static extern void clear(IntPtr model);

    public Transform[] trainingSpheres;
    public Transform[] testSpheres;

    private double[] trainingInputs;
    private double[] trainingOutputs;

    private int[] layout;

    private IntPtr model;

    public void ReInitialize()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                0f,
                testSpheres[i].position.z);
        }
    }
    
    public void CreateModel()
    {
        ReleaseModel();
        layout = new int[2] { 2, 1 };
        model = createPCMModel(layout, layout.Length);
        //PredictOnTestSpheres();
    }

    public void Train()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingOutputs = new double[trainingSpheres.Length];

        for (int i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
            trainingOutputs[i] = trainingSpheres[i].position.y;
        }
        
        trainPCMClassification(model, trainingInputs, trainingOutputs, trainingSpheres.Length, 10000, 0.0001);

    }

    public void PredictOnTestSpheres()
    {
        for (int i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            var predictedY = predictPCMClassification(model, input);
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                Convert.ToSingle(predictedY),
                testSpheres[i].position.z);
        }
    }

    public void ReleaseModel()
    {
        if (model != null)
        {
            clear(model);
            ReInitialize();
            model = IntPtr.Zero;
        }
    }
}