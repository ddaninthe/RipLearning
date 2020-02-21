using System;
using System.Runtime.InteropServices;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class MLPNonSepLinear2D50EMulticlassScript : MonoBehaviour
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
    private static extern void trainMLPClassification(IntPtr model, double[] dataset, double[] expectedOutputs, int datasetSize, double nbIter, double learning);

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
            testSpheres[i].gameObject.GetComponent<MeshRenderer>().material.color = Color.white;
        }
    }
    
    public void CreateModel()
    {
        ReleaseModel();
        layout = new int[2] { 2, 3 };
        model = createMLPModel(layout, layout.Length);
        //PredictOnTestSpheres();
    }

    public void Train()
    {
        string trainingSphereElement;
        string instanceString = " (Instance)";
        int expectedLength = layout[layout.Length - 1];
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingOutputs = new double[trainingSpheres.Length * expectedLength];

        for (int i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;

            trainingSphereElement = trainingSpheres[i].gameObject.GetComponent<MeshRenderer>().material.name;
            Debug.Log("training spheres type numero " + i + " = " + trainingSphereElement);
            for (int j = 0; j < expectedLength; j++) {
                trainingOutputs[expectedLength * i + j] = -1.0;
            }
            if (trainingSphereElement == "red" + instanceString) { trainingOutputs[expectedLength * i] = 1.0; }
            else if (trainingSphereElement == "blue" + instanceString) { trainingOutputs[expectedLength * i + 1] = 1.0; }
            else { trainingOutputs[expectedLength * i + 2] = 1.0; }

        }

        for (int i = 0; i < trainingOutputs.Length; i++)
        {
            Debug.Log("training Output numero " + i + " : " + trainingOutputs[i]);
        }
        

        trainMLPClassification(model, trainingInputs, trainingOutputs, trainingSpheres.Length, 10000, 0.01);

    }

    public void PredictOnTestSpheres()
    {
        int expectedLength = layout[layout.Length - 1];
        for (int i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] { testSpheres[i].position.x, testSpheres[i].position.z };
            Debug.Log("sphere " + i + " : x = " + testSpheres[i].position.x + " / z = " + testSpheres[i].position.z);
            double[] predictedYArray = new double[expectedLength];
            Marshal.Copy(predictMLPClassification(model, input), predictedYArray, 0, expectedLength);

            for (int j = 0; j < predictedYArray.Length; j++)
            {
                Debug.Log("predictedArray " + i + " item " + j + " -> " + predictedYArray[j]);
            }
            

            double maxValue = predictedYArray.Max();
            int predictedY = predictedYArray.ToList().IndexOf(maxValue);
            Debug.Log(" Pour sphere " + i + " maxValue =  " + maxValue + " A l'index : " + predictedY);
            if (predictedY == 0) { testSpheres[i].gameObject.GetComponent<MeshRenderer>().material.color = Color.red; }
            else if (predictedY == 1) { testSpheres[i].gameObject.GetComponent<MeshRenderer>().material.color = Color.blue; }
            else { testSpheres[i].gameObject.GetComponent<MeshRenderer>().material.color = Color.yellow; }
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