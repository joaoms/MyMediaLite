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
using System.IO;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;

class WSDMCupTask2
{
	string method = "ISGD";
	int random_seed = 10;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	List<Tuple<int,int>> test_data;
	string submission_filename;
	StreamWriter submission_file, log_file;


	public WSDMCupTask2(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: wsdm_cup_task2 <recommender> <\"recommender params\"> <train_file> <test_file> <submission_file>[ <random_seed>]");
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
		test_data = ReadTestData(args[3]);
		submission_filename = args[4];

		if(args.Length > 5) random_seed = Int32.Parse(args[5]);
		MyMediaLite.Random.Seed = random_seed;

		DateTime dt = DateTime.Now;

		log_file = new StreamWriter("log" + String.Format("{0:yyMMddHHmmss}", dt));

	}

	public static void Main(string[] args)
	{
		var program = new WSDMCupTask2(args);
		program.Run();
	}

	private List<Tuple<int,int>> ReadTestData(string filename)
	{
		var ret = new List<Tuple<int,int>>();
		var reader = new StreamReader(filename);

		string line;
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(Constants.SPLIT_CHARS);

			if (tokens.Length < 2)
				throw new FormatException("Expected at least 2 columns: " + line);

			try
			{
				int user_id = user_mapping.ToInternalID(tokens[0]);
				int item_id = item_mapping.ToInternalID(tokens[1]);
				ret.Add(Tuple.Create(user_id, item_id));
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}
		return ret;

	}

	private void Run()
	{
		var candidate_items = train_data.AllItems;
		var predictions = new List<double>(test_data.Count);

		recommender.Feedback = train_data;

		log_file.WriteLine(recommender.ToString());

		DateTime start_train = DateTime.Now;
		recommender.Train();
		TimeSpan train_time = DateTime.Now - start_train;

		log_file.WriteLine("Train time: " + train_time.TotalMilliseconds);

		for (int i = 0; i < test_data.Count; i++)
		{
			int tu = test_data[i].Item1;
			int ti = test_data[i].Item2;
			log_file.WriteLine("\n" + tu + " " + ti + "\n");
			predictions.Add(recommender.Predict(tu, ti));

			if(i % (test_data.Count/100) == 0)
			{
				Console.Write(".");
				GC.Collect();
			}
		}
		Console.WriteLine("Writing submission file...");
		submission_file = new StreamWriter(submission_filename + String.Format("{0:yyMMddHHmmss}", DateTime.Now));
		submission_file.WriteLine("id,target");
		for(int i = 0; i < predictions.Count; i++)
			submission_file.WriteLine(i + "," + predictions[i]);
		submission_file.Close();
	}


	private void SetupRecommender(string parameters)
	{
		recommender.Configure(parameters);
	}


}
