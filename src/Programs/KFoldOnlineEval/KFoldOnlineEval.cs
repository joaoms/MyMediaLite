// Copyright (C) 2013 Zeno Gantner
//
// This file is part of MyMediaLite.
//
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
//
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;
using MyMediaLite.Eval.Measures;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

class KFoldOnlineEval
{
	string method = "ISGD";
	string parameters = "";
	int random_seed = 10;
	int n_recs = 10;
	int n_folds = 10;
	string fold_type = "bootstrap";
	bool repeated_items = false;
	IncrementalItemRecommender[] recommenders;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	readonly IDictionary<string, IList<double>> measures;
	readonly string [] metrics = { "recall_1", "recall_5", "recall_10", "recall_20", "map", "auc", "ndcg", "time" };
	int output_interval = 1000;

	string[,] output_buffer;
	int[] output_buffer_count;
	StreamWriter output_info;
	StreamWriter[] output;

	public KFoldOnlineEval(string[] args)
	{
		if(args.Length < 6) {
			Console.WriteLine("Usage: kfold_online_eval <recommender> <\"recommender params\"> <training_file> <test_file> [<n_folds> [<fold_type> [<random_seed> [<n_recs> [<repeated_items> [<output_interval>]]]]]]");
			Environment.Exit(1);
		}

		method = args[0];
		if(method.CreateItemRecommender() == null)
		{	
			Console.WriteLine("Invalid method: " + method);
			Environment.Exit(1);
		}

		parameters = args[1];

		train_data = ItemData.Read(args[2], user_mapping, item_mapping);
		test_data = ItemData.Read(args[3], user_mapping, item_mapping);

		if(args.Length > 4) n_folds = int.Parse(args[4]);

		if(args.Length > 5) fold_type = args[5];

		if(args.Length > 6) random_seed = int.Parse(args[6]);
		MyMediaLite.Random.Seed = random_seed;
		
		if(args.Length > 7) n_recs = int.Parse(args[7]);

		if(args.Length > 8) repeated_items = bool.Parse(args[8]);

		if(args.Length > 9) output_interval = int.Parse(args[9]);

		measures = InitMeasures();

		DateTime dt = DateTime.Now;

		var dir = string.Format("{0:yyMMddHHmmss}", dt);
		Directory.CreateDirectory(dir);
		output = new StreamWriter[metrics.Count()];

		output_buffer = new string[metrics.Count(), output_interval];
		output_buffer_count = Enumerable.Repeat(0, metrics.Count()).ToArray();

		output_info = new StreamWriter(dir + "/info" + method +
		                               args[3].Substring(args[3].LastIndexOf("/", StringComparison.Ordinal)+1) + ".log");
		for (int i = 0; i < metrics.Count(); i++)
		{
			output[i] = new StreamWriter(dir + "/" + metrics[i] + method + 
			                             args[3].Substring(args[3].LastIndexOf("/", StringComparison.Ordinal)+1) + ".log");
			output[i].Write("idx\tuser\titem");
			for (int j = 0; j < n_folds; j++)
				output[i].Write("\tfold_" + j);
			output[i].WriteLine();
		}
	}

	public static void Main(string[] args)
	{
		var program = new KFoldOnlineEval(args);
		program.Run();
	}
	
	private void Run()
	{

		SetupRecommenders(method, parameters);

		InitRecommenders();



		for (int i = 0; i < test_data.Count; i++)
		{
			EvaluatePrequential(i);

			WriteOutputBuffer();
			if(i % 5000 == 0)
				GC.Collect();
		}

		Terminate();

	}

	private void EvaluatePrequential(int index)
	{
		bool recommend = false;
		Tuple<int,int> tuple;
		HashSet<int> ignore_items;

		DateTime instance_start, instance_end;
		double total_time;

		var user = test_data.Users[index];
		var item = test_data.Items[index];



		var rand = new System.Random(random_seed);
		var train_test = new int[n_folds];



		var results = new Dictionary<string, double[]>(metrics.Count());
		foreach (var metric in metrics)
			results.Add(metric, new double[n_folds]);

		var candidate_items = new List<int>(train_data.AllItems.Union(test_data.AllItems));

		if ("bootstrap".Equals(fold_type, StringComparison.OrdinalIgnoreCase))
		{
			for (int i = 0; i < n_folds; i++)
				train_test[i] = Poisson.Sample(rand, 1);
		} 
		else if ("split".Equals(fold_type, StringComparison.OrdinalIgnoreCase))
		{
			train_test = Enumerable.Repeat(0, n_folds).ToArray();
			var k = rand.Next(0, n_folds - 1);
			train_test[k] = 1;
		} 
		else
		{
			train_test = Enumerable.Repeat(1, n_folds).ToArray();
			var k = rand.Next(0, n_folds - 1);
			train_test[k] = 0;
		}


		if (train_data.AllUsers.Contains(user))
		{
			recommend = true;
			if(!repeated_items)
				if (train_data.UserMatrix[user].Contains(item))
					recommend = false;
		}

		instance_start = DateTime.Now;
		Parallel.For(0, n_folds, i => {	

			DateTime fold_start, fold_end;
			IList<Tuple<int,float>> rec_list_score;
			List<int> rec_list,til;

			fold_start = DateTime.Now;

			if(recommend)
			{
				if(repeated_items)
					ignore_items = new HashSet<int>();
				else
					ignore_items = new HashSet<int>(train_data.UserMatrix[user]);

				rec_list_score = recommenders[i].Recommend(user, n_recs, ignore_items, candidate_items);
				rec_list = new List<int>();
				foreach (var rec in rec_list_score)
					rec_list.Add(rec.Item1);

				til = new List<int>(){ item };

				results["recall_1"][i] = PrecisionAndRecall.RecallAt(rec_list, til, 1);
				results["recall_5"][i] = PrecisionAndRecall.RecallAt(rec_list, til, 5);
				results["recall_10"][i] = PrecisionAndRecall.RecallAt(rec_list, til, 10);
				results["recall_20"][i] = PrecisionAndRecall.RecallAt(rec_list, til, 20);
				results["map"][i] = PrecisionAndRecall.AP(rec_list, til);
				var num_dropped = candidate_items.Count - ignore_items.Count - n_recs;
				results["auc"][i] = AUC.Compute(rec_list, til, num_dropped);
				results["ndcg"][i] = NDCG.Compute(rec_list, til);
			}

			// update recommenders
			tuple = Tuple.Create(user, item);
			if (train_test[i] == 0)
			{
				recommenders[i].UpdateUsers = false;
				recommenders[i].UpdateItems = false;
				recommenders[i].AddFeedback(new Tuple<int, int>[]{ tuple });
				recommenders[i].UpdateUsers = true;
				recommenders[i].UpdateItems = true;
			} else {
				var tuples = Enumerable.Repeat(tuple, train_test[i]).ToArray();
				recommenders[i].AddFeedback(tuples);
			}

			fold_end = DateTime.Now;
			results["time"][i] = (fold_end - fold_start).TotalMilliseconds;

		});
		instance_end = DateTime.Now;
		total_time = (instance_end - instance_start).TotalMilliseconds;

		if (recommend)
		{
			measures["recall_1"].Add(results["recall_1"].Average());
			measures["recall_5"].Add(results["recall_5"].Average());
			measures["recall_10"].Add(results["recall_10"].Average());
			measures["recall_20"].Add(results["recall_20"].Average());
			measures["map"].Add(results["map"].Average());
			measures["auc"].Add(results["auc"].Average());
			measures["ndcg"].Add(results["ndcg"].Average());
			for (int j = 0; j < metrics.Count() - 1; j++)
			{
				output_buffer[j, output_buffer_count[j]] = index + "\t" + user_mapping.ToOriginalID(user) + "\t" + item_mapping.ToOriginalID(item);
				for (int k = 0; k < n_folds; k++)
					output_buffer[j, output_buffer_count[j]] += "\t" + results[metrics[j]][k];
				output_buffer[j, output_buffer_count[j]] += "\t" + results[metrics[j]].Average();
				output_buffer_count[j]++;
			}
		}
		measures["time"].Add(results["time"].Average());
		var l = metrics.Count() - 1;
		output_buffer[l, output_buffer_count[l]] = index + "\t" + user_mapping.ToOriginalID(user) + "\t" + item_mapping.ToOriginalID(item);
		for (int k = 0; k < n_folds; k++)
			output_buffer[l, output_buffer_count[l]] += "\t" + results["time"][k];
		output_buffer[l, output_buffer_count[l]] += "\t" + results["time"].Average();
		output_buffer[l, output_buffer_count[l]] += "\t" + results["time"].Sum();
	}

	private void WriteOutputBuffer(bool final = false) {
		for (int l = 0; l < metrics.Count(); l++)
		{
			if (output_buffer_count[l] == output_interval || final)
			{
				for (int i = 0; i < output_buffer_count[l]; i++)
					output[l].WriteLine(output_buffer[l, i]);
				output_buffer_count[l] = 0;
			}
		}
	}

	private void SetupRecommenders(string alg, string parameters)
	{
		recommenders = new IncrementalItemRecommender[n_folds];
		for (int i = 0; i < n_folds; i++)
		{
			recommenders[i] = (IncrementalItemRecommender) alg.CreateItemRecommender();
			recommenders[i].Configure(parameters);
		}
	}

	private void InitRecommenders()
	{
		DateTime start_train = DateTime.Now;
		for (int i = 0; i < n_folds; i++)
			output_info.WriteLine(recommenders[0].ToString());
		Parallel.ForEach(recommenders, recommender => {
			recommender.Feedback = train_data;
			recommender.Train();
			recommender.UpdateUsers = false;
			recommender.UpdateItems = false;
		});
		TimeSpan train_time = DateTime.Now - start_train;
		for (int i = 0; i < n_folds; i++)
			output_info.WriteLine("Train time: " + train_time.TotalMilliseconds);
	}

	private IDictionary<string, IList<double>> InitMeasures()
	{
		var dict = new Dictionary<string, IList<double>>();
		foreach (var metric in metrics)
			dict.Add(metric, new List<double>());
		dict.Add("retrain_times", new List<double>());
		return dict;
	}

	private void Terminate()
	{

		WriteOutputBuffer(true);
		//Compute and print averages
		for (int i = 0; i < metrics.Count(); i++)
			output[i].WriteLine();
		foreach (var measure in measures)
		{
			double score = Math.Round(measure.Value.Average(), 5);
			for (int i = 0; i < n_folds; i++)
				output_info.WriteLine(measure.Key + ":\t" + score);
	
		}

		for (int i = 0; i < metrics.Count(); i++)
			output[i].Close();
		output_info.Close();
	}

}
