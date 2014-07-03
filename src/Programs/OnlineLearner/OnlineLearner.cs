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

class OnlineLearner
{
	string method = "SimpleSGD";
	int random_seed = 10;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	string model_filename;
	string usermap_filename;
	string itemmap_filename;


	public OnlineLearner(string[] args)
	{
		if(args.Length < 6) {
			Console.WriteLine("Usage: online_learner <recommender> <\"recommender params\"> <training_file> <model_file> <user_map> <item_map> [<random_seed>]");
			Environment.Exit(1);
		}

		method = args[0];
		recommender = (IncrementalItemRecommender) method.CreateItemRecommender();
		if (recommender == null) {
			Console.WriteLine("Invalid method: "+method);
			Environment.Exit(1);
		}

		recommender.Configure(args[1]);

		train_data = ItemData.Read(args[2], user_mapping, item_mapping);

		model_filename = args[3];
		usermap_filename = args[4];
		itemmap_filename = args[5];

		if(args.Length > 6) random_seed = Int32.Parse(args[6]);
		MyMediaLite.Random.Seed = random_seed;

		user_mapping.SaveMapping(usermap_filename);
		item_mapping.SaveMapping(itemmap_filename);
	}

	public static void Main(string[] args)
	{
		var program = new OnlineLearner(args);
		program.Run();
	}

	private void Run()
	{
		Tuple<int,int> tuple;
		List<TimeSpan> update_times = new List<TimeSpan>(1000);

		recommender.Feedback.Add(train_data.Users[0],train_data.Items[0]);

		Console.WriteLine(recommender.ToString());

		recommender.Train();
		Console.WriteLine();

		DateTime retrain_start, retrain_end;

		for (int i = 1; i < train_data.Count; i++)
		{
			// update recommender
			tuple = Tuple.Create(train_data.Users[i], train_data.Items[i]);
			retrain_start = DateTime.Now;
			recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
			retrain_end = DateTime.Now;
			update_times.Add(retrain_end-retrain_start);

			if(i % 1000 == 0)
			{
				Console.WriteLine(update_times.Average(x => x.TotalMilliseconds));
				if(i % 5000 == 0)
					System.GC.Collect();
			}
		}

		Console.WriteLine("Saving model to " + model_filename + " ...");
		recommender.SaveModel(model_filename);
		Console.WriteLine("done.");
	
	}
		
}
