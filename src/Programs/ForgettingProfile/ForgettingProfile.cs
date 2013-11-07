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
using System.Collections.Concurrent;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;

class ForgettingProfile
{
	string method = "NaiveSVD";
	int random_seed = 10;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	IList<float[]> scores;
	ParallelOptions parallel_opts = new ParallelOptions() { MaxDegreeOfParallelism = 4 };

	public ForgettingProfile(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: forgetting_profile <recommender> <\"recommender params\"> <training_file> <test_file> [<random_seed>]");
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

	}

	public static void Main(string[] args)
	{
		var program = new ForgettingProfile(args);
		program.Run();
	}

	private void Run()
	{

		scores = new List<float[]>(test_data.Count + 1);

		recommender.Feedback = train_data;

		Console.WriteLine(recommender.ToString());

		recommender.Train();

		var s = new float[train_data.Count];

		for (int i = 0; i < train_data.Count; i++)
			s[i] = recommender.Predict(train_data.Users[i], train_data.Items[i]);

		scores.Add(s);

		Console.WriteLine("Trained model with " + train_data.Count + " observations");

		for (int i = 0; i < test_data.Count; i++)
		{
			int tu = test_data.Users[i];
			int ti = test_data.Items[i];

			var pair = Tuple.Create(tu, ti);
			recommender.AddFeedback(new Tuple<int, int>[]{ pair });

			var sc = new float[recommender.Feedback.Count];

			Parallel.For (0, recommender.Feedback.Count, parallel_opts, j => {
				sc[j] = recommender.Predict(recommender.Feedback.Users[j], 
				                            recommender.Feedback.Items[j]);
			});

			scores.Add(sc);

			if(i % 5000 == 0)
			{
				Console.Write(".");
				System.GC.Collect();
			}
		}

		Terminate();

	}

	private void SetupRecommender(string parameters)
	{
		recommender.Configure(parameters);
	}

	private void Terminate()
	{
		DateTime dt = DateTime.Now;
		string filename = "forgetting_profiles_" + method + String.Format("{0:yyMMddHHmm}", dt) + ".log";
		Console.WriteLine("Writing data in: " + filename);
		// Save results
		StreamWriter w = new StreamWriter(filename);
		for (int i = 0; i < recommender.Feedback.Count; i++)
		{
			var user = recommender.Feedback.Users[i];
			var item = recommender.Feedback.Items[i];

			w.Write(user + "-" + item);

			foreach(var s in scores[i])
				w.Write("\t" + s);
			w.WriteLine();
		}

		w.Close();
	}

}
