using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using Accord.MachineLearning.DecisionTrees;
using Accord.Math.Optimization.Losses;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning;
using Accord.Statistics.Analysis;

namespace PokerHandClass
{
    class Program
    {
        public static int iterations = 1;
        static void Main(string[] args)
        {
            Dirbam();

        }
        static void Dirbam()
        {
            DownloadTrainingAndTestingData();
            List<int[]> trainingData = ReadData("poker-hand-training-true.data");
            List<int[]> testingData = ReadData("poker-hand-testing.data");
            double[] metoduTikslumai = new Double[2];
           // metoduTikslumai[0] = RandomForestClassification(trainingData, testingData);
            //metoduTikslumai[1] = DecisionTreeClassification(trainingData, testingData);
            metoduTikslumai[2] = kNearestNeighbours(trainingData, testingData);

           // double[] ss = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 };
            //double[] prob = new double[3];
            //RandomForest ranForest = RandomForestClassification(trainingData, testingData, out prob[0]);
            //Console.WriteLine(ranForest.Decide(ss));

            //DecisionTree decisionTree = DecisionTreeClassification(trainingData, testingData, out prob[1]);
            //Console.WriteLine(decisionTree.Decide(ss));
            //Console.ReadKey();
        }
        static RandomForest RandomForestClassification(List<int[]> trainingData, List<int[]> testingData, out double prob)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            double prec = 0;
            Console.WriteLine("Random Forest Classification");
            RandomForest bestforest = null;
            for (int i = 0; i < iterations; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing from: {0} to {1}", indexTestingStart, indexTestingEnd);
                int[][] inputData, testinputData;
                int[] outputData, testoutputData;

                PrepareInputOutput(out inputData, out outputData, out testinputData, out testoutputData, trainingData, testingData, indexTestingStart, indexTestingEnd);
                var RanForest = new RandomForestLearning()
                {
                    NumberOfTrees = 5,
                };
                var forest = RanForest.Learn(inputData, outputData);
                Console.WriteLine("Medis sukurtas - ismokta");
                //int[] predicted = forest.Decide(inputData);
                //int[] predicTest = forest.Decide(testinputData);
                double er = new ZeroOneLoss(testoutputData).Loss(forest.Decide(testinputData));
                Console.WriteLine("Apmokymo tikslumas: {0}", 1 - er);
                if(1 - er > prec)
                {
                    prec = 1 - er;
                    bestforest = forest;
                }
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                errorAverage += er;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
            prob = 1 - (errorAverage / iterations);
            return bestforest;
        }
        static DecisionTree DecisionTreeClassification(List<int[]> trainingData, List<int[]> testingData, out double prob)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            double prec = 0;
            Console.WriteLine("Decision Tree Classification");
            DecisionTree bestDecision = null;
            for (int i = 0; i < iterations; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing from: {0} to {1}", indexTestingStart, indexTestingEnd);
                int[][] inputData, testinputData;
                int[] outputData, testoutputData;

                PrepareInputOutput(out inputData, out outputData, out testinputData, out testoutputData, trainingData, testingData, indexTestingStart, indexTestingEnd);

                ID3Learning teacher = new ID3Learning();
                var decision = teacher.Learn(inputData, outputData);
                Console.WriteLine("Medis sukurtas - ismokta");
                double error = new ZeroOneLoss(testoutputData).Loss(decision.Decide(testinputData));
                Console.WriteLine("Apmokymo tikslumas: {0}", 1 - error);
                if (1 - error > prec)
                {
                    prec = 1 - error;
                    bestDecision = decision;
                }
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                errorAverage += error;
                bestDecision = decision;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
            prob = 1 - (errorAverage / iterations);
            return bestDecision;
        }
        static double kNearestNeighbours(List<int[]> trainingData, List<int[]> testingData)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            Console.WriteLine("k nearest neighbours Classification");
            for (int i = 0; i < 10; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing nuo: {0} iki {1}", indexTestingStart, indexTestingEnd);
                int[][] inputData, testinputData;
                int[] outputData, testoutputData;
                PrepareInputOutput(out inputData, out outputData, out testinputData, out testoutputData, trainingData, testingData, indexTestingStart, indexTestingEnd);
                double[][] input = new double[inputData.GetLength(0)][];
                double a = 0;
                for (int j = 0; j < inputData.GetLength(0); j++)
                {
                    input[j] = new double[10];
                    for(int k = 0; k < 10; k++)
                    {
                        a = Convert.ToDouble(inputData[j][k]);
                        input[j][k] = a;
                    }
                }
                var knn = new KNearestNeighbors(k: 4);
                knn.Learn(input, outputData);
                var cm = GeneralConfusionMatrix.Estimate(knn, input, outputData);
                // We can use it to estimate measures such as 
                double error = cm.Error;  // should be 0
                double acc = cm.Accuracy; // should be 1
                double kappa = cm.Kappa;  // should be 1
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
            }
            return 0;
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
