﻿using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class RegressionScript : MonoBehaviour
{

    [DllImport("ViDLL.dll")]
    private static extern IntPtr createLinearModel(int nbInputs);

    [DllImport("ViDLL.dll")]
    private static extern IntPtr trainLinearRegression(double[] dataset, int dataSize, IntPtr model, int modelSize, double iterNumber, double learning);

    [DllImport("ViDLL.dll")]
    private static extern double predictLinearRegression(IntPtr model, int size, double[] inputs);

    [DllImport("ViDLL.dll")]
    private static extern void clear(IntPtr model);

    public Transform[] trainingSpheres;
    public Transform[] testSpheres;

    private double[] trainingInputs;

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
        model = createLinearModel(2);
        PredictOnTestSpheres();
    }

    public void Train()
    {
        trainingInputs = new double[trainingSpheres.Length * 3];

        for (int i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[3 * i] = trainingSpheres[i].position.x;
            trainingInputs[3 * i + 1] = trainingSpheres[i].position.z;
            trainingInputs[3 * i + 2] = trainingSpheres[i].position.y;
        }
        
        trainLinearRegression(trainingInputs, trainingInputs.Length, model, 2, 100000, 0.0001);
        
    }

    public void PredictOnTestSpheres()
    {
        for (int i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] { testSpheres[i].position.x, testSpheres[i].position.z };
            var predictedY = predictLinearRegression(model, 2, input);
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