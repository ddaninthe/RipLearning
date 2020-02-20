using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class TransformNonLinearCROSSScript : MonoBehaviour
{

    [DllImport("ViDLL.dll")]
    private static extern IntPtr createLinearModel(int nbInputs);

    [DllImport("ViDLL.dll")]
    private static extern IntPtr trainLinearClassification(double[] dataset, int datasetSize, double[] expectedOutputs, IntPtr model, int modelSize, double nbIter, double learning);

    [DllImport("ViDLL.dll")]
    private static extern int predictLinearClassification(IntPtr model, int size, double[] inputs);

    [DllImport("ViDLL.dll")]
    private static extern void clear(IntPtr model);

    public Transform[] trainingSpheres;
    public Transform[] testSpheres;

    private double[] trainingInputs;
    private double[] trainingOutputs;

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
        model = createLinearModel(3);
        /*for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingSpheres[i].position = new Vector3(
                Math.Abs(trainingSpheres[i].position.x),
                trainingSpheres[i].position.y,
                Math.Abs(trainingSpheres[i].position.z));
        }*/
        //PredictOnTestSpheres();
    }

    public void Train()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingOutputs = new double[trainingSpheres.Length];

        for (int i = 0; i < trainingSpheres.Length; i++)
        {
            //trainingInputs[2 * i] = Math.Abs(trainingSpheres[i].position.x);
            //trainingInputs[2 * i + 1] = Math.Abs(trainingSpheres[i].position.z);
            /*if ((trainingSpheres[i].position.x < 0 && trainingSpheres[i].position.z < 0) || (trainingSpheres[i].position.x >= 0 && trainingSpheres[i].position.z >= 0))
            {
                trainingOutputs[i] = trainingSpheres[i].position.y * -1;
            }
            else trainingOutputs[i] = trainingSpheres[i].position.y;*/
            trainingInputs[2 * i] = trainingSpheres[i].position.y;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.y;
            trainingOutputs[i] = trainingSpheres[i].position.y;
        }

        trainLinearClassification(trainingInputs, trainingSpheres.Length, trainingOutputs, model, 2, 10000, 0.0001);

    }

    public void PredictOnTestSpheres()
    {
        for (int i = 0; i < testSpheres.Length; i++)
        {
            double inputx, inputz;
            inputx = inputz = (testSpheres[i].position.x >= -2 && testSpheres[i].position.x <= 2) || (testSpheres[i].position.z >= -2 && testSpheres[i].position.z <= 2) ? 1 : -1;
            var input = new double[] { inputx, inputz };
            var predictedY = predictLinearClassification(model, 2, input);
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