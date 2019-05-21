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
            double[] ss = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 }; // Cia testavimui, reikes is consoles paimt input ir tada prognozuoja geriausias metodas is musu triju.


            double[] prob = new double[3];
            RandomForest ranForest = RandomForestClassification(trainingData, testingData, out prob[0]);
            Console.WriteLine(ranForest.Decide(ss));

            double Tomo = DecisionTreeClassification(trainingData, testingData);
            double Tado = 0.0; // Tado metodas

        }
        static RandomForest RandomForestClassification(List<int[]> trainingData, List<int[]> testingData, out double prob)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            Console.WriteLine("Random Forest Classification");
            RandomForest bestforest = null;
            for (int i = 0; i < iterations; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing nuo: {0} iki {1}", indexTestingStart, indexTestingEnd);
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
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                errorAverage += er;
                bestforest = forest;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
            prob = 1 - (errorAverage / iterations);
            return bestforest;
        }
        static double DecisionTreeClassification(List<int[]> trainingData, List<int[]> testingData)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;
            double errorAverage = 0;
            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            Console.WriteLine("Decision Tree Classification");
            for (int i = 0; i < iterations; i++)
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
            return 1 - (errorAverage / iterations);
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
