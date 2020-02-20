using System;
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
    private static extern IntPtr createMLPModel(int[] layout, int arraySize);

    [DllImport("ViDLL.dll")]
    private static extern void trainMLPClassification(IntPtr model, double[] dataset, double[] expectedOutputs, int datasetSize, int iteractions, double alpha);

    [DllImport("ViDLL.dll")]
    private static extern IntPtr predictMLPClassification(IntPtr model, double[] data);

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
        model = createMLPModel(layout, layout.Length);
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

        trainMLPClassification(model, trainingInputs, trainingOutputs, trainingSpheres.Length, 100, 0.01);

    }

    public void PredictOnTestSpheres()
    {
        int expectedLength = layout[layout.Length - 1];
        for (int i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            double[] predictedYArray = new double[expectedLength];
            Marshal.Copy(predictMLPClassification(model, input), predictedYArray, 0, expectedLength);
            Debug.Log("predictedArray first item -> " + predictedYArray[0]);
            double predictedY = predictedYArray[0];
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                Convert.ToSingle(predictedY = predictedY >= 0.0 ? 1.0 : -1.0),
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