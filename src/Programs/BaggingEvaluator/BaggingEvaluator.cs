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
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;

class BaggingEvaluator
{
	string method = "NaiveSVD";
	string recommender_params;
	int random_seed = 10;
	int n_recs = 10;
	int num_bags = 10;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	IList<int> candidate_items;
	ParallelOptions parallel_opts = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

	public BaggingEvaluator(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: online_evaluator <recommender> <\"recommender params\"> <training_file> <test_file> <num_bags> [<random_seed> [<n_recs>]]");
			Environment.Exit(1);
		}
		
		method = args[0];
		recommender_params = args[1];

		train_data = ItemData.Read(args[2], user_mapping, item_mapping);
		test_data = ItemData.Read(args[3], user_mapping, item_mapping);

		if(args.Length > 4) num_bags = Int32.Parse(args[4]);

		if(args.Length > 5) random_seed = Int32.Parse(args[5]);
		MyMediaLite.Random.Seed = random_seed;
		
		if(args.Length > 6) n_recs = Int32.Parse(args[6]);

	}

	public static void Main(string[] args)
	{
		var program = new BaggingEvaluator(args);
		program.Run();
	}
	
	private void Run()
	{

		var train_bags = CreateTrainBags();
		List<List<int>> test_idbags = CreateTestBagIds();

		Parallel.For(0, num_bags, parallel_opts, this_bag => {

			DateTime rec_start, rec_end, retrain_start, retrain_end;
			var log = new StreamWriter("ov" + method + "_bag" + this_bag + ".log");

			var candidate_items = new List<int>(train_data.AllItems.Union(test_data.AllItems));
			var measures = InitMeasures();

			var recommender = (IncrementalItemRecommender) method.CreateItemRecommender();
			if (recommender == null) {
				Console.WriteLine("Invalid method: "+method);
				Environment.Exit(1);
			}
		
			SetupRecommender(recommender, recommender_params);
			recommender.SetProperty(param_name, param_val);
			recommender.Feedback = train_bags[this_bag];

			log.WriteLine(recommender.ToString());

			var train_start = DateTime.Now;
			recommender.Train();
			var train_time = DateTime.Now - train_start;
			log.WriteLine("Train time: " + train_time.TotalMilliseconds);

			for (int i = 0; i < test_data.Count; i++)
			{
				int tu = test_data.Users[i];
				int ti = test_data.Items[i];
				log.WriteLine("\n" + tu + " " + ti);
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
							log.WriteLine(rec.Item1);
							rec_list.Add(rec.Item1);
						}

						var til = new List<int>(){ ti };
						int num_dropped = candidate_items.Count - ignore_items.Count - n_recs;
						measures["recall@1"].Add(
							MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 1));
						measures["recall@5"].Add(
							MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 5));
						measures["recall@10"].Add(
							MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 10));
						measures["MAP"].Add(
							MyMediaLite.Eval.Measures.PrecisionAndRecall.AP(rec_list, til));
						measures["AUC"].Add(
							MyMediaLite.Eval.Measures.AUC.Compute(rec_list, til, num_dropped));
						measures["NDCG"].Add(
							MyMediaLite.Eval.Measures.NDCG.Compute(rec_list, til));
						measures["rec_time"].Add((rec_end - rec_start).TotalMilliseconds);
					}
				}
				// update recommender
				retrain_start = DateTime.Now;
				if(test_idbags[this_bag].Contains(i)) {
					var tuple = Tuple.Create(tu, ti);
					recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
				} else {
					// TODO: this removes the item from candidates to *all users*
					// need to create ignore_items by user structure
					candidate_items.RemoveAll(lambda => lambda == ti);
				}
				retrain_end = DateTime.Now;
				measures["retrain_time"].Add((retrain_end - retrain_start).TotalMilliseconds);
				if(i % 5000 == 0)
					System.GC.Collect();
			}

			WriteResults(this_bag, log, measures);

			log.Close();

		});

	}

	private void SetupRecommender(Recommender rec, string parameters)
	{
		rec.Configure(parameters);
	}

	private IList<IPosOnlyFeedback> CreateTrainBags()
	{
		double logistic = 1 - 1 / Math.E;
		var rand = MyMediaLite.Random.GetInstance();
		var train_bags = new List<PosOnlyFeedback<SparseBooleanMatrix>>();
		for (int i = 0; i < num_bags; i++)
		{
			var bag = new PosOnlyFeedback<SparseBooleanMatrix>(); 
			for (int j = 0; j < train_data.Count; j++)
			{
				double rn = 0.001 * rand.Next(0,1000); 
				if(rn < logistic) 
					bag.Add(train_data.Users[j], train_data.Items[j]);
			}
			train_bags.Add(bag);
		}
		return bags;
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

	private void WriteResults(int num_bag, StreamWriter log, IDictionary<string, IList<double>> measures)
	{
		int count = measures["retrain_time"].Count;

		//Compute and print averages
		log.WriteLine();
		foreach (var measure in measures)
		{
			double score = Math.Round(measure.Value.Average(), 5);
			log.WriteLine(measure.Key + ":\t" + score);
	
		}

		// Save one by one results
		StreamWriter w = new StreamWriter("measures" + method + "_bag" + num_bag + ".log");
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

		w = new StreamWriter("retrain_times" + method + "_bag" + num_bag + ".log");

		foreach (var t in measures["retrain_time"])
			w.WriteLine(t);

		w.Close();
	}

}
