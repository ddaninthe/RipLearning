using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class RBFClassificationScript : MonoBehaviour
{

    [DllImport("ViDLL.dll")]
    private static extern IntPtr createRBFModel(double[] dataset, int datasetSize, int dimensions);

    [DllImport("ViDLL.dll")]
    private static extern IntPtr trainNaiveRBF(IntPtr model, int datasetSize, double[] expectedOutputs, int dimensions, double gamma);

    [DllImport("ViDLL.dll")]
    private static extern int predictRBFClassification(IntPtr model, double gamma, double[] inputs, int dimensions, int modelSize);

    [DllImport("ViDLL.dll")]
    private static extern void clear(IntPtr model);

    public Transform[] trainingSpheres;
    public Transform[] testSpheres;

    private double[] trainingInputs;
    private double[] trainingOutputs;

    private double gamma = 10.0;
    private int dimensions = 2;

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

        trainingInputs = new double[trainingSpheres.Length * 2];

        for (int i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
        }

        model = createRBFModel(trainingInputs, trainingSpheres.Length, 2);
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
        
        trainNaiveRBF(model, trainingSpheres.Length, trainingOutputs, dimensions, gamma);

    }

    public void PredictOnTestSpheres()
    {
        for (int i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            var predictedY = predictRBFClassification(model, gamma, input, dimensions, trainingSpheres.Length);
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