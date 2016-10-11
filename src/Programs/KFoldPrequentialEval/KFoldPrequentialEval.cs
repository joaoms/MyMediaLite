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
using MyMediaLite.DataType;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;
using MyMediaLite.Eval.Measures;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

class KFoldPrequentialEval
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
	IPosOnlyFeedback all_train_data;
	IPosOnlyFeedback all_test_data;
	IPosOnlyFeedback[] fold_train_data;
	IPosOnlyFeedback[] fold_test_data;
	IDictionary<int, int[]> user_folds;
	Dictionary<string, IList<double>[]> results;
	readonly string [] metrics = { "recall_1", "recall_5", "recall_10", "recall_20", "map", "auc", "ndcg", "rec_time" };
	int output_interval = 1000;

	string[,] output_buffer;
	int[] output_buffer_count;
	string[,] output_buffer_time;
	int[] output_buffer_count_time;
	StreamWriter output_info;
	StreamWriter[] output, output_time;

	public KFoldPrequentialEval(string[] args)
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

		all_train_data = ItemData.Read(args[2], user_mapping, item_mapping);
		all_test_data = ItemData.Read(args[3], user_mapping, item_mapping);

		SplitFoldData();

		if(args.Length > 4) n_folds = int.Parse(args[4]);

		if(args.Length > 5) fold_type = args[5];

		if(args.Length > 6) random_seed = int.Parse(args[6]);
		MyMediaLite.Random.Seed = random_seed;

		if(args.Length > 7) n_recs = int.Parse(args[7]);

		if(args.Length > 8) repeated_items = bool.Parse(args[8]);

		if(args.Length > 9) output_interval = int.Parse(args[9]);

		results = InitResults();

		DateTime dt = DateTime.Now;

		var dir = string.Format("{0:yyMMddHHmmss}", dt);
		Directory.CreateDirectory(dir);
		output = new StreamWriter[n_folds];
		output_time = new StreamWriter[n_folds];

		output_buffer = new string[n_folds, output_interval];
		output_buffer_count = Enumerable.Repeat(0, n_folds).ToArray();

		output_buffer_time = new string[n_folds, output_interval];
		output_buffer_count = Enumerable.Repeat(0, n_folds).ToArray();

		output_info = new StreamWriter(dir + "/info" + method +
		                               args[3].Substring(args[3].LastIndexOf("/", StringComparison.Ordinal)+1) + ".log");

		for (int f = 0; f < n_folds; f++)
		{
			output[f] = new StreamWriter(dir + "/" + "fold" + f.ToString("D2") + method + 
			                             args[3].Substring(args[3].LastIndexOf("/", StringComparison.Ordinal)+1) + ".log");
			output_time[f] = new StreamWriter(dir + "/time" + "fold" + f.ToString("D2") + method + 
			                             args[3].Substring(args[3].LastIndexOf("/", StringComparison.Ordinal)+1) + ".log");
			output[f].Write("idx\tuser\titem");
			output_time[f].Write("idx\tuser\titem");

			for (int j = 0; j < metrics.Count(); j++)
				output[f].Write("\t" + metrics[j]);
			output[f].WriteLine();
			output_time[f].WriteLine("\tupd_time");
		}
	}

	public static void Main(string[] args)
	{
		var program = new KFoldPrequentialEval(args);
		program.Run();
	}

	private void Run()
	{

		SetupRecommenders(method, parameters);
		InitRecommenders();
		EvaluatePrequential();

		Terminate();

	}

	private void SplitFoldData()
	{
		var all_users = new HashSet<int>(all_train_data.AllUsers.Union(all_test_data.AllUsers));
		user_folds = new Dictionary<int, int[]>(all_users.Count);
		var rand = new System.Random(random_seed);
		int[] folds;
		int k;
		foreach (int user in all_users)
		{
			switch(fold_type)
			{
				case "split": // split-validation
				folds = Enumerable.Repeat(0, n_folds).ToArray();
				k = rand.Next(0, n_folds - 1);
				folds[k] = 1;
				break;

				case "cv": // cross-validation
				folds = Enumerable.Repeat(1, n_folds).ToArray();
				k = rand.Next(0, n_folds - 1);
				folds[k] = 0;
				break;

				default: // bootstrap
				folds = new int[n_folds];
				for (k = 0; k < n_folds; k++)
					folds[k] = Poisson.Sample(rand, 1);
				break;
			}
			user_folds.Add(user, folds);
		}

		fold_train_data = new PosOnlyFeedback<SparseBooleanMatrix>[n_folds];
		fold_test_data = new PosOnlyFeedback<SparseBooleanMatrix>[n_folds];

		for (int f = 0; f < n_folds; f++)
		{
			fold_train_data[f] = new PosOnlyFeedback<SparseBooleanMatrix>();
			fold_test_data[f] = new PosOnlyFeedback<SparseBooleanMatrix>();
		}

		for (int i = 0; i < all_train_data.Count; i++)
		{
			int user_id = all_train_data.Users[i];
			int item_id = all_train_data.Items[i];

			folds = user_folds[user_id];
			for (int f = 0; f < n_folds; f++)
				for (k = 0; k < folds[f]; k++)
					fold_train_data[f].Add(user_id, item_id);
		}

		for (int i = 0; i < all_test_data.Count; i++)
		{
			int user_id = all_test_data.Users[i];
			int item_id = all_test_data.Items[i];

			folds = user_folds[user_id];
			for (int f = 0; f < n_folds; f++)
				for (k = 0; k < folds[f]; k++)
					fold_test_data[f].Add(user_id, item_id);
		}

	}

	private void EvaluatePrequential()
	{

		Parallel.For(0, n_folds, f => {
			var candidate_items = new List<int>(fold_train_data[f].AllItems.Union(fold_test_data[f].AllItems));
			bool recommend = false;
			DateTime train_start, train_end;
			IList<Tuple<int,float>> rec_list_score;
			List<int> rec_list,til;
			ISet<int> ignore_items;
			Tuple<int,int> tuple;

			for (int i = 0; i < fold_test_data[f].Count; i++)
			{
				int user = fold_test_data[f].Users[i];
				int item = fold_test_data[f].Items[i];

				if (fold_train_data[f].AllUsers.Contains(user))
				{
					recommend = true;
					if (!repeated_items)
						recommend &= !fold_train_data[f].UserMatrix[user].Contains(item);
				}
		
				if(recommend)
				{
					if(repeated_items)
						ignore_items = new HashSet<int>();
					else
						ignore_items = new HashSet<int>(fold_train_data[f].UserMatrix[user]);

					var rec_start = DateTime.Now;
					rec_list_score = recommenders[f].Recommend(user, n_recs, ignore_items, candidate_items);
					var rec_end = DateTime.Now;
					rec_list = new List<int>();
					foreach (var rec in rec_list_score)
						rec_list.Add(rec.Item1);

					til = new List<int>(){ item };

					results["recall_1"][f].Add(PrecisionAndRecall.RecallAt(rec_list, til, 1));
					results["recall_5"][f].Add(PrecisionAndRecall.RecallAt(rec_list, til, 5));
					results["recall_10"][f].Add(PrecisionAndRecall.RecallAt(rec_list, til, 10));
					results["recall_20"][f].Add(PrecisionAndRecall.RecallAt(rec_list, til, 20));
					results["map"][f].Add(PrecisionAndRecall.AP(rec_list, til));
					var num_dropped = candidate_items.Count - ignore_items.Count - n_recs;
					results["auc"][f].Add(AUC.Compute(rec_list, til, num_dropped));
					results["ndcg"][f].Add(NDCG.Compute(rec_list, til));
					results["rec_time"][f].Add((rec_end - rec_start).TotalMilliseconds);
				
					output_buffer[f, output_buffer_count[f]] = i + "\t" + user_mapping.ToOriginalID(user) + "\t" + item_mapping.ToOriginalID(item);
					for (int k = 0; k < metrics.Count(); k++)
						output_buffer[f, output_buffer_count[f]] += "\t" + results[metrics[k]][f].Last();
					output_buffer_count[f]++;
				}

				// update recommenders
				train_start = DateTime.Now;

				tuple = Tuple.Create(user, item);
				recommenders[i].AddFeedback(new Tuple<int, int>[]{ tuple });

				train_end = DateTime.Now;
				results["upd_time"][f].Add((train_end - train_start).TotalMilliseconds);

				output_buffer_time[f, output_buffer_count_time[f]] = i + "\t" + user_mapping.ToOriginalID(user) + "\t" 
				                                                                            + item_mapping.ToOriginalID(item);
				output_buffer_time[f, output_buffer_count_time[f]] += "\t" + results["upd_time"][f].Last();
				output_buffer_count_time[f]++;

				WriteOutputBuffer();
				if(i % 5000 == 0)
					GC.Collect();
			}
		});
	}

	private void WriteOutputBuffer(bool final = false) {
		for (int f = 0; f < n_folds; f++)
		{
			if (output_buffer_count[f] == output_interval || final)
			{
				for (int i = 0; i < output_buffer_count[f]; i++)
					output[f].WriteLine(output_buffer[f, i]);
			output_buffer_count[f] = 0;
			}
			if (output_buffer_count_time[f] == output_interval || final)
			{
				for (int i = 0; i < output_buffer_count_time[f]; i++)
					output_time[f].WriteLine(output_buffer_time[f, i]);
				output_buffer_count_time[f] = 0;
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
		{
			output_info.WriteLine(recommenders[0].ToString());
			recommenders[i].Feedback = fold_train_data[i];
		}

		Parallel.ForEach(recommenders, recommender => {
			recommender.Train();
		});

		TimeSpan train_time = DateTime.Now - start_train;

		for (int i = 0; i < n_folds; i++)
			output_info.WriteLine("Train time: " + train_time.TotalMilliseconds);
	}

	private Dictionary<string, IList<double>[]> InitResults()
	{
		var res = new Dictionary<string, IList<double>[]>(metrics.Count());
		foreach (var metric in metrics)
		{
			res.Add(metric, new List<double>[n_folds]);
			for (int f = 0; f < n_folds; f++)
				res[metric][f] = new List<double>();
		}
		res.Add("upd_time", new List<double>[n_folds]);
		for (int f = 0; f < n_folds; f++)
			res["upd_time"][f] = new List<double>();
		return res;
	}

	private void Terminate()
	{
		double score_sum = 0;
		WriteOutputBuffer(true);
		//Compute and print averages
		output_info.WriteLine();
		output_info.Write("metric");
		for (int f = 0; f < n_folds; f++)
		{
			output[f].WriteLine();
			output_info.Write("\tfold " + f.ToString("D2"));
		}
		output_info.WriteLine("\tavg");

		foreach (var result in results)
		{
			output_info.Write(result.Key);
			for (int f = 0; f < n_folds; f++)
			{
				double score = result.Value[f].Average();
				score_sum += score;
				output_info.Write("\t" + Math.Round(score, 5));
			}
			output_info.WriteLine("\t" + Math.Round(score_sum/n_folds, 5));
		}

		for (int i = 0; i < metrics.Count(); i++)
			output[i].Close();
		output_info.Close();
	}

}
