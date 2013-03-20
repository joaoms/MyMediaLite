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

class OnlineCrossval
{
	string method = "NaiveSVD";
	int random_seed = 10;
	int n_recs = 10;
	//float split = 0.2f;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback all_data;
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	IDictionary<string, IList<double>> measures;

	public OnlineCrossval(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: online_crossval <recommender> <\"recommender params\"> <data_file> <split> [<random_seed> [<n_recs>]]");
			Environment.Exit(1);
		}
		
		method = args[0];
		recommender = (IncrementalItemRecommender) method.CreateItemRecommender();
		if (recommender == null) {
			Console.WriteLine("Invalid method: "+method);
			Environment.Exit(1);
		}
		
		SetupRecommender(args[1]);
		
		all_data = ItemData.Read(args[2], user_mapping, item_mapping);

		SplitData(args[3]);

		if(args.Length > 4) random_seed = Int32.Parse(args[4]);
		MyMediaLite.Random.Seed = random_seed;
		
		if(args.Length > 5) n_recs = Int32.Parse(args[5]);

		measures = InitMeasures();

	}

	public static void Main(string[] args)
	{
		var program = new OnlineCrossval(args);
		program.Run();
	}
	
	private void Run()
	{

		var candidate_items = new List<int>(all_data.AllItems);

		recommender.Feedback = train_data;

		//Console.WriteLine(recommender.ToString());

		recommender.Train();

		for (int i = 0; i < test_data.Count; i++)
		{
			int tu = test_data.Users[i];
			int ti = test_data.Items[i];
			if (train_data.AllUsers.Contains(tu))
			{
				if(!train_data.UserMatrix[tu].Contains(ti))
				{

					var ignore_items = new HashSet<int>(train_data.UserMatrix[tu]);
					var rec_list_score = recommender.Recommend(tu, n_recs, ignore_items, candidate_items);
					var rec_list = new List<int>();
					foreach (var rec in rec_list_score)
					{
						//Console.WriteLine(rec.Item1);
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
				}
			}
			// update recommender
			var tuple = Tuple.Create(tu, ti);
			recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
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
		return dict;
	}

	private void Terminate()
	{
		//string[] metrics = new string[]{"recall@1","recall@5","recall@10","MAP","AUC","NDCG"};
		//int count = measures[metrics[0]].Count;
		//DateTime dt = DateTime.Now;

		//Compute and print averages
		//Console.WriteLine();
		foreach (var measure in measures)
		{
			double score = measure.Value.Sum() / measure.Value.Count;
			Console.Write(score + "\t");
		}
		Console.WriteLine();

		// Save one by one results
		/*
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
		*/
	}

	private void SplitData(string split)
	{
		double spl = -1;
		try 
		{
			spl = Double.Parse(split, System.Globalization.CultureInfo.InvariantCulture);
		} 
		catch(Exception) 
		{
			Console.WriteLine("Split value invalid!");
			Environment.Exit(1);
		}		
		if(spl <= 0 || spl >=1) 
		{
			Console.WriteLine("Split value must be between 0 and 1 (excl)!");
			Environment.Exit(1);
		}

		train_data = new PosOnlyFeedback<SparseBooleanMatrix>();
		test_data = new PosOnlyFeedback<SparseBooleanMatrix>();

		int cnt = all_data.Count;
		int split_idx = (int) Math.Ceiling(cnt * spl);
		//Console.WriteLine("Split point and index " + split_idx);
		for (int i = 0; i < cnt; i++)
		{
			if(i < split_idx) train_data.Add(all_data.Users[i], all_data.Items[i]);
			else test_data.Add(all_data.Users[i], all_data.Items[i]);
		}
		//Console.WriteLine("train data: " + train_data.Count + " / test data: " + test_data.Count);
	}

}
