using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;

namespace TPTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            MLContext mlContext = new MLContext(seed: 0);

            List<Microsoft.ML.Data.TextLoader.Column> mlCols = new List<Microsoft.ML.Data.TextLoader.Column>();

            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric1", Microsoft.ML.Data.DataKind.Single, 0));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric10", Microsoft.ML.Data.DataKind.Single, 1));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric2", Microsoft.ML.Data.DataKind.Single, 2));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric3", Microsoft.ML.Data.DataKind.Single, 3));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric4", Microsoft.ML.Data.DataKind.Single, 4));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric5", Microsoft.ML.Data.DataKind.Single, 5));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric6", Microsoft.ML.Data.DataKind.Single, 6));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric7", Microsoft.ML.Data.DataKind.Single, 7));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric8", Microsoft.ML.Data.DataKind.Single, 8));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Metric9", Microsoft.ML.Data.DataKind.Single, 9));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Stage1Timing", Microsoft.ML.Data.DataKind.Single, 10));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Stage2Timing", Microsoft.ML.Data.DataKind.Single, 11));
            mlCols.Add(new Microsoft.ML.Data.TextLoader.Column("Stage3Timing", Microsoft.ML.Data.DataKind.Single, 12));

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile(Directory.GetCurrentDirectory() + "//Files//training.csv", mlCols.ToArray(), ',', true, true, true, false);
            IDataView testingDataView = mlContext.Data.LoadFromTextFile(Directory.GetCurrentDirectory() + "//Files//testing.csv", mlCols.ToArray(), ',', true, true, true, false);
            IEstimator<ITransformer> pipeline = mlContext.Transforms.CopyColumns("Label", "Stage3Timing");
            List<string> concatCols = new List<string>();

            concatCols.Add("Metric1");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric1"));
            concatCols.Add("Metric10");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric10"));
            concatCols.Add("Metric2");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric2"));
            concatCols.Add("Metric3");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric3"));
            concatCols.Add("Metric4");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric4"));
            concatCols.Add("Metric5");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric5"));
            concatCols.Add("Metric6");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric6"));
            concatCols.Add("Metric7");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric7"));
            concatCols.Add("Metric8");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric8"));
            concatCols.Add("Metric9");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Metric9"));
            concatCols.Add("Stage1Timing");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Stage1Timing"));
            concatCols.Add("Stage2Timing");
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Stage2Timing"));

            pipeline = pipeline.Append(mlContext.Transforms.Concatenate("Features", concatCols.ToArray()));

            var trainer = mlContext.Regression.Trainers.Sdca(new Microsoft.ML.Trainers.SdcaRegressionTrainer.Options()
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            });

            var trainingPipeline = pipeline.Append(trainer);

            DateTime dtTrainStart = DateTime.Now;

            Console.WriteLine("Starting Training: " + dtTrainStart.ToString());

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("Finished Training: " + DateTime.Now.ToString());
            Console.WriteLine("Took " + Math.Abs((dtTrainStart - DateTime.Now).TotalSeconds) + " Seconds");

            IDataView predictions = trainedModel.Transform(testingDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine($"*Metrics for {trainer.ToString()} regression model");
            Console.WriteLine(string.Empty);
            Console.WriteLine($"*LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*RMS loss:      {metrics.RootMeanSquaredError:#.##}");

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, Directory.GetCurrentDirectory() + "//Files//Model.zip");

            //mlContext.Data.CreateEnumerable()

            ////Run Prediction
            //DataViewSchema schema;
            //ITransformer trainedModelLoaded = mlContext.Model.Load(new MemoryStream(File.ReadAllBytes(Directory.GetCurrentDirectory() + "//Files//Model.zip")), out schema);

            //ModelOperationsCatalog modelOperationsCatalog = new ModelOperationsCatalog(new Type[] { });

            //PredictionEngine predictionEngine = modelOperationsCatalog.CreatePredictionEngine(trainedModelLoaded, schema);

        }
    }
}
