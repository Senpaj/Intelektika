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

namespace PokerHandClass
{
    class Program
    {
        static void Main(string[] args)
        {
            DownloadTrainingAndTestingData();
            List<int[]> trainingData = ReadData("poker-hand-training-true.data");
            List<int[]> testingData = ReadData("poker-hand-testing.data");
            RandomForestClassification(trainingData, testingData);

        }
        static void RandomForestClassification(List<int[]> trainingData, List<int[]> testingData)
        {
            int testingCount = testingData.Count / 10;
            int trainingCount = testingData.Count - testingCount;

            int indexTestingStart = testingData.Count - testingCount;
            int indexTestingEnd = testingData.Count;
            for (int i = 0; i < 10; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Console.WriteLine("Testing nuo: {0} iki {1}", indexTestingStart, indexTestingEnd);
                List<int[]> a = GetTrainingFiles(testingData, indexTestingStart, indexTestingEnd);
                List<int[]> b = GetTestingFiles(testingData, indexTestingStart, indexTestingEnd);
                int[][] inputData = new int[a.Count][];
                int[] outputData = new int[a.Count];
                for (int ii = 0; ii < a.Count; ii++)
                {
                    inputData[ii] = new int[10];
                    for (int j = 0; j < 10; j++)
                    {
                        inputData[ii][j] = a[ii][j];
                    }
                    outputData[ii] = a[ii][10];
                }
                int[][] testinputData = new int[b.Count][];
                int[] testoutputData = new int[b.Count];
                for (int ii = 0; ii < b.Count; ii++)
                {
                    testinputData[ii] = new int[10];
                    for (int j = 0; j < 10; j++)
                    {
                        testinputData[ii][j] = b[ii][j];
                    }
                    testoutputData[ii] = b[ii][10];
                }
                var teacher = new RandomForestLearning()
                {
                    NumberOfTrees = 100,
                };
                var forest = teacher.Learn(inputData, outputData);
                Console.WriteLine("Medis sukurtas - ismokta");
                //int[] predicted = forest.Decide(inputData);
                //int[] predicTest = forest.Decide(testinputData);
                double er = new ZeroOneLoss(testoutputData).Loss(forest.Decide(testinputData));
                Console.WriteLine("Apmokymo tikslumas: {0}", 1-er);
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Iteracija baigta per: {0}ms", elapsedMs);
                indexTestingEnd = indexTestingStart;
                indexTestingStart -= testingCount;
                Console.WriteLine("------------------------------------------------------------------------------");
            }
        }
        static List<int[]> ReadData(string file)
        {
            List<int[]> temp = new List<int[]>();
            using (StreamReader r = new StreamReader(file))
            {
                string line;
                while(null != (line = r.ReadLine()))
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
            Console.WriteLine(temp.Count);
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
