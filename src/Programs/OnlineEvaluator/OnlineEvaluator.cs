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

		recommender.Train();

		for (int i = 0; i < test_data.Count; i++) {
			int tu = test_data.Users[i];
			int ti = test_data.Items[i];
			Console.WriteLine("\n" + tu + " " + ti);
			if (train_data.AllUsers.Contains(tu))
			{
				var ignore_items = new HashSet<int>(train_data.UserMatrix[tu]);
				var rec_list = recommender.Recommend(tu, n_recs, ignore_items, candidate_items);
				foreach (var rec in rec_list)
					Console.WriteLine(rec.Item1);
			}
			// update recommender
			var tuple = Tuple.Create(tu, ti);
			recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
			if(i % 10000 == 0)
				System.GC.Collect();
		}

	}

	private void SetupRecommender(string parameters) {
		recommender.Configure(parameters);
	}
}
