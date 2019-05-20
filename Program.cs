using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.InputOutput.Csv;
using SharpLearning.RandomForest;
using Accord.MachineLearning.DecisionTrees;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning.DecisionTrees.Learning;

namespace PokerHandClass
{
    class Program
    {
        static void Main(string[] args)
        {
            DownloadTrainingAndTestingData();
            List<int[]> trainingData = ReadData("poker-hand-training-true.data");
            List<int[]> testingData = ReadData("poker-hand-testing.data");
            double[] metoduTikslumai = new Double[2];
            metoduTikslumai[0] = RandomForestClassification(trainingData, testingData);
            metoduTikslumai[1] = DecisionTreeClassification(trainingData, testingData);
            metoduTikslumai[2] = 0.0; // Tado metodas

        }
        static double RandomForestClassification(List<int[]> trainingData, List<int[]> testingData)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            Console.WriteLine("Random Forest Classification");
            for (int i = 0; i < 10; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing nuo: {0} iki {1}", indexTestingStart, indexTestingEnd);
                int[][] inputData, testinputData;
                int[] outputData, testoutputData;

                PrepareInputOutput(out inputData, out outputData, out testinputData, out testoutputData, trainingData, testingData, indexTestingStart, indexTestingEnd);

                var teacher = new RandomForestLearning()
                {
                    NumberOfTrees = 100,
                };
                var forest = teacher.Learn(inputData, outputData);
                Console.WriteLine("Medis sukurtas - ismokta");
                //int[] predicted = forest.Decide(inputData);
                //int[] predicTest = forest.Decide(testinputData);
                double er = new ZeroOneLoss(testoutputData).Loss(forest.Decide(testinputData));
                Console.WriteLine("Apmokymo tikslumas: {0}", 1 - er);
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                errorAverage += er;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
            return 1 - (errorAverage / 10);
        }
        static double DecisionTreeClassification(List<int[]> trainingData, List<int[]> testingData)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            Console.WriteLine("Decision Tree Classification");
            for (int i = 0; i < 10; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing nuo: {0} iki {1}", indexTestingStart, indexTestingEnd);
                int[][] inputData, testinputData;
                int[] outputData, testoutputData;

                PrepareInputOutput(out inputData, out outputData, out testinputData, out testoutputData, trainingData, testingData, indexTestingStart, indexTestingEnd);

                ID3Learning teacher = new ID3Learning();
                var tree = teacher.Learn(inputData, outputData);
                Console.WriteLine("Medis sukurtas - ismokta");
                //int[] predicted = forest.Decide(inputData);
                //int[] predicTest = forest.Decide(testinputData);
                double error = new ZeroOneLoss(testoutputData).Loss(tree.Decide(testinputData));
                Console.WriteLine("Apmokymo tikslumas: {0}", 1 - error);
                int[] predicted = tree.Decide(inputData);
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                errorAverage += error;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
            return 1 - (errorAverage / 10);
        }
        static void PrepareInputOutput(out int[][] inputData, out int[] outputData, out int[][] testinputData, out int[] testoutputData, List<int[]> trainingData, List<int[]> testingData, int indexTestingStart, int indexTestingEnd)
        {
            List<int[]> a = GetTrainingFiles(testingData, indexTestingStart, indexTestingEnd);
            List<int[]> b = GetTestingFiles(testingData, indexTestingStart, indexTestingEnd);
            inputData = new int[a.Count][];
            outputData = new int[a.Count];
            for (int ii = 0; ii < a.Count; ii++)
            {
                inputData[ii] = new int[10];
                for (int j = 0; j < 10; j++)
                {
                    inputData[ii][j] = a[ii][j];
                }
                outputData[ii] = a[ii][10];
            }
            testinputData = new int[b.Count][];
            testoutputData = new int[b.Count];
            for (int ii = 0; ii < b.Count; ii++)
            {
                testinputData[ii] = new int[10];
                for (int j = 0; j < 10; j++)
                {
                    testinputData[ii][j] = b[ii][j];
                }
                testoutputData[ii] = b[ii][10];
            }
        }
        static List<int[]> ReadData(string file)
        {
            List<int[]> temp = new List<int[]>();
            using (StreamReader r = new StreamReader(file))
            {
                string line;
                while (null != (line = r.ReadLine()))
                {
                    string[] val = line.Split(',');
                    int[] arr = new int[11];
                    for (int i = 0; i < val.Length; i++)
                    {
                        arr[i] = Convert.ToInt32(val[i]);
                    }
                    temp.Add(arr);
                }
            }
            return temp;
        }
        static void DownloadTrainingAndTestingData()
        {
            if (!File.Exists("poker-hand-training-true.data"))
            {
                Console.WriteLine("The training file does not exists. Attempting to download it!");
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data",
                        "poker-hand-training-true.data");
                }
            }
            if (!File.Exists("poker-hand-testing.data"))
            {
                Console.WriteLine("The testing file does not exists. Attempting to download it!");
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data",
                        "poker-hand-testing.data");
                }
            }
        }
        static List<int[]> GetTestingFiles(List<int[]> files, int indexStart, int indexEnd)
        {
            List<int[]> obj = new List<int[]>();
            for (int i = indexStart; i < indexEnd; i++)
            {
                obj.Add(files[i]);
            }
            return obj;
        }
        static List<int[]> GetTrainingFiles(List<int[]> files, int indexStart, int indexEnd)
        {
            List<int[]> obj = new List<int[]>();
            for (int i = 0; i < indexStart; i++)
            {
                obj.Add(files[i]);
            }
            for (int i = indexEnd; i < files.Count; i++)
            {
                obj.Add(files[i]);
            }
            return obj;
        }
    }
}
