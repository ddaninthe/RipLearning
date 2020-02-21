using System;
using System.IO;
using System.Runtime.InteropServices;
using Random = UnityEngine.Random;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Dataset : MonoBehaviour
{
    [DllImport("ViDLL.dll")]
    private static extern IntPtr createLinearModel(int nbInputs);

    [DllImport("ViDLL.dll")]
    private static extern IntPtr trainLinearClassification(double[] dataset, int datasetSize, double[] expectedOutputs, IntPtr model, int modelSize, double nbIter, double learning);

    [DllImport("ViDLL.dll")]
    private static extern int predictLinearClassification(IntPtr model, int size, double[] inputs);

    [DllImport("ViDLL.dll")]
    private static extern void clear(IntPtr model);

    private int[] layout;


    public IntPtr CreateLinearModel(int nbInput)
    {
        IntPtr model = createLinearModel(nbInput);
        //PredictOnTestSpheres();
        return model;
    }

    public (double[,] data, string[] expect, int length, int size) openData()
    {
        double[,] data = new double[151,4];
        string[] expect = new string[151];
        var lines = File.ReadAllLines(@"Assets\iris.data");
        var i = 0;
        int length = 0;
        foreach (var line in lines)
        {
            char[] spearator = { '|', ' ' }; 
		    Int32 count = 5;
		    string[] splitted = line.Split(spearator, count, StringSplitOptions.None);

            
            for (int a = 0; a < splitted.Length - 1; a++)
            {
                data[i, a] = Convert.ToDouble(splitted[a]);
            }

            length = splitted.Length - 1;
            expect[i] = splitted[splitted.Length - 1];
            i++;
        }
        return (data, expect, i, length);
    }

    public void TrainLinear(IntPtr model, double[] data, double[] expect, int length, int size)
    {
        trainLinearClassification(data, length, expect, model, size, 10000, 0.0001);
    }

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello world");

        var raw = openData();

        int dataLength = raw.length;
        int dataSize = 4;

        var expect = raw.expect;
        var dataRaw = raw.data;

        double[] data = new double[dataLength * dataSize];

        int count = 0;
        for (int i = 0; i < dataLength; i++)
        {
            for (int j = 0; j < dataSize; j++)
            {
                data[count] = dataRaw[i, j];
                count++;
            }
        }


        double[] expectS = new double[150];
        double[] expectV = new double[150];
        double[] expectG = new double[150];

        //setup setosa
        for (int i = 0; i < 150; i++)
        {
            if (expect[i] == "Iris-setosa")
            {
                expectS[i] = 1.0;
            } else
            {
                expectS[i] = -1.0;
            }
        }

        //setup versicolor
        for (int i = 0; i < 150; i++)
        {
            if (expect[i] == "Iris-versicolor")
            {
                expectV[i] = 1.0;
            }
            else
            {
                expectV[i] = -1.0;
            }
        }

        //setup virginica
        for (int i = 0; i < 150; i++)
        {
            if (expect[i] == "Iris-virginica")
            {
                expectG[i] = 1.0;
            }
            else
            {
                expectG[i] = -1.0;
            }
        }

        int[] layout = { 4, 1 };

        var modelS = CreateLinearModel(dataLength);
        var modelV = CreateLinearModel(dataLength);
        var modelG = CreateLinearModel(dataLength);

        TrainLinear(modelS, data, expectS, dataLength, dataSize);
        TrainLinear(modelV, data, expectV, dataLength, dataSize);
        TrainLinear(modelG, data, expectG, dataLength, dataSize);

        //double[] test = { 5.1, 3.5, 1.4, 0.2 };
        //double[] test = { 6.6, 2.9, 4.6, 1.3 };
        //double[] test = { 6.7, 3.3, 5.7, 2.1 };

        double[] test = { 4.2, 6.6, 7.8, 9.7 };

        double predictS = predictLinearClassification(modelS, dataSize, test);
        double predictV = predictLinearClassification(modelV, dataSize, test);
        double predictG = predictLinearClassification(modelG, dataSize, test);

        Debug.Log(predictS);
        Debug.Log(predictV);
        Debug.Log(predictG);
    }
}