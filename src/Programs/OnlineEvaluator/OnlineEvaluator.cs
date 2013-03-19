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
using System.Collections;
using System.Linq;
using System.IO;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;

class OnlineEvaluator
{
	string method = "NaiveSVD";
	int random_seed = 10;
	int n_recs = 10;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	IDictionary<string, IList<double>> measures;

	public OnlineEvaluator(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: online_evaluator <recommender> <\"recommender params\"> <training_file> <test_file> [<random_seed> [<n_recs>]]");
			Environment.Exit(1);
		}
		
		method = args[0];
		recommender = (IncrementalItemRecommender) method.CreateItemRecommender();
		if (recommender == null) {
			Console.WriteLine("Invalid method: "+method);
			Environment.Exit(1);
		}
		
		SetupRecommender(args[1]);
		
		train_data = ItemData.Read(args[2], user_mapping, item_mapping);
		test_data = ItemData.Read(args[3], user_mapping, item_mapping);

		if(args.Length > 4) random_seed = Int32.Parse(args[4]);
		MyMediaLite.Random.Seed = random_seed;
		
		if(args.Length > 5) n_recs = Int32.Parse(args[5]);

		measures = InitMeasures();

	}

	public static void Main(string[] args)
	{
		var program = new OnlineEvaluator(args);
		program.Run();
	}
	
	private void Run()
	{

		var candidate_items = new List<int>(train_data.AllItems.Union(test_data.AllItems));

		recommender.Feedback = train_data;

		Console.WriteLine(recommender.ToString());

		DateTime start_train = DateTime.Now;
		recommender.Train();
		TimeSpan train_time = DateTime.Now - start_train;

		Console.WriteLine("Train time: " + train_time.TotalMilliseconds);

		DateTime rec_start, rec_end, retrain_start, retrain_end;

		for (int i = 0; i < test_data.Count; i++)
		{
			int tu = test_data.Users[i];
			int ti = test_data.Items[i];
			Console.WriteLine("\n" + tu + " " + ti);
			if (train_data.AllUsers.Contains(tu))
			{
				if(!train_data.UserMatrix[tu].Contains(ti))
				{

					var ignore_items = new HashSet<int>(train_data.UserMatrix[tu]);
					rec_start = DateTime.Now;
					var rec_list_score = recommender.Recommend(tu, n_recs, ignore_items, candidate_items);
					rec_end = DateTime.Now;
					var rec_list = new List<int>();
					foreach (var rec in rec_list_score)
					{
						Console.WriteLine(rec.Item1);
						rec_list.Add(rec.Item1);
					}

					var til = new List<int>(){ ti };
					measures["recall@1"].Add(
						MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 1));
					measures["recall@5"].Add(
						MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 5));
					measures["recall@10"].Add(
						MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 10));
					measures["MAP"].Add(
						MyMediaLite.Eval.Measures.PrecisionAndRecall.AP(rec_list, til));
					int num_dropped = candidate_items.Count - ignore_items.Count - n_recs;
					measures["AUC"].Add(
						MyMediaLite.Eval.Measures.AUC.Compute(rec_list, til, num_dropped));
					measures["NDCG"].Add(
						MyMediaLite.Eval.Measures.NDCG.Compute(rec_list, til));
					measures["rec_time"].Add((rec_end - rec_start).TotalMilliseconds);
				}
			}
			// update recommender
			var tuple = Tuple.Create(tu, ti);
			retrain_start = DateTime.Now;
			recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
			retrain_end = DateTime.Now;
			measures["retrain_time"].Add((retrain_end - retrain_start).TotalMilliseconds);
			if(i % 5000 == 0)
				System.GC.Collect();
		}

		Terminate();

	}

	private void SetupRecommender(string parameters)
	{
		recommender.Configure(parameters);
	}

	private IDictionary<string, IList<double>> InitMeasures()
	{
		var dict = new Dictionary<string, IList<double>>();
		dict.Add("recall@1", new List<double>());
		dict.Add("recall@5", new List<double>());
		dict.Add("recall@10", new List<double>());
		dict.Add("MAP", new List<double>());
		dict.Add("AUC", new List<double>());
		dict.Add("NDCG", new List<double>());
		dict.Add("rec_time", new List<double>());
		dict.Add("retrain_time", new List<double>());
		return dict;
	}

	private void Terminate()
	{
		string[] metrics = new string[]{"recall@1","recall@5","recall@10","MAP","AUC","NDCG","rec_time"};
		int count = measures[metrics[0]].Count;
		DateTime dt = DateTime.Now;

		//Compute and print averages
		Console.WriteLine();
		foreach (var measure in measures)
		{
			double score = measure.Value.Sum() / measure.Value.Count;
			Console.WriteLine(measure.Key + ":\t" + score);
	
		}

		// Save one by one results
		StreamWriter w = new StreamWriter("measures" + String.Format("{0:yyMMddHHmm}", dt) + ".log");
		foreach (string m in metrics)
			w.Write(m + "\t");
		w.WriteLine();
		for (int i = 0; i < count; i++)
		{
			foreach (string m in metrics)
				w.Write(measures[m][i] + "\t");
			w.WriteLine();
		}

		w.Close();

		w = new StreamWriter("retrain_times" + String.Format("{0:yyMMddHHmm}", dt) + ".log");

		foreach (var t in measures["retrain_time"])
			w.WriteLine(t);

		w.Close();
	}

}
