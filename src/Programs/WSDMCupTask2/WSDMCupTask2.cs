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
	string method = "ISGDWSDM";
	int random_seed = 10;
	ISGDWSDM recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	Tuple<List<Tuple<int,int>>,List<int>> train_data_r;
	List<Tuple<int,int>> test_data;
	string submission_filename;
	StreamWriter submission_file, log_file;
	DateTime dt = DateTime.Now;


	public WSDMCupTask2(string[] args)
	{
		if(args.Length < 5) {
			Console.WriteLine("Usage: wsdm_cup_task2 <recommender> <\"recommender params\"> <train_file> <test_file> <submission_file>[ <random_seed>]");
			Environment.Exit(1);
		}

		log_file = new StreamWriter("log" + string.Format("{0:yyMMddHHmmss}", dt));

		/*
		method = args[0];
		recommender = (IncrementalItemRecommender) method.CreateItemRecommender();
		if (recommender == null) {
			Console.WriteLine("Invalid method: "+method);
			Environment.Exit(1);
		}
		*/
		recommender = new ISGDWSDM();

		Console.WriteLine("Configuring recommender " + method);
		SetupRecommender(args[1]);
		log_file.WriteLine(recommender.ToString());

		Console.WriteLine("Reading train data...");
		ReadTrainData(args[2]);

		Console.WriteLine("Reading test data...");
		ReadTestData(args[3]);
		submission_filename = args[4];

		if(args.Length > 5) random_seed = int.Parse(args[5]);
		MyMediaLite.Random.Seed = random_seed;



	}

	public static void Main(string[] args)
	{
		var program = new WSDMCupTask2(args);
		program.Run();
	}

	private void ReadTrainData(string filename)
	{
		var ret_ui = new List<Tuple<int,int>>();
		var ret_r = new List<int>();
		var reader = new StreamReader(filename);

		string line = reader.ReadLine();
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(Constants.SPLIT_CHARS);

			if (tokens.Length < 6)
				throw new FormatException("Expected at least 6 columns: " + line);

			try
			{
				int user_id = user_mapping.ToInternalID(tokens[0]);
				int item_id = item_mapping.ToInternalID(tokens[1]);
				int rating = int.Parse(tokens[5]);
				recommender.Feedback.Add(user_id, item_id);
				recommender.AddScore(rating);
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}

	}

	private void ReadTestData(string filename)
	{
		test_data = new List<Tuple<int, int>>();
		var reader = new StreamReader(filename);

		string line = reader.ReadLine();
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(Constants.SPLIT_CHARS);

			if (tokens.Length < 2)
				throw new FormatException("Expected at least 2 columns: " + line);

			try
			{
				int user_id = user_mapping.ToInternalID(tokens[1]);
				int item_id = item_mapping.ToInternalID(tokens[2]);
				test_data.Add(Tuple.Create(user_id, item_id));
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}

	}

	private void Run()
	{
		var candidate_items = recommender.Feedback.AllItems;
		var predictions = new List<double>(test_data.Count);

		Console.WriteLine("Training...");

		DateTime start_train = DateTime.Now;
		recommender.Train();
		TimeSpan train_time = DateTime.Now - start_train;

		Console.WriteLine("Train time: " + train_time.TotalMilliseconds);

		for (int i = 0; i < test_data.Count; i++)
		{
			int tu = test_data[i].Item1;
			int ti = test_data[i].Item2;
			log_file.WriteLine(tu + " " + ti);
			predictions.Add(Math.Max(Math.Min(recommender.Predict(tu, ti), 1d), 0d));

			if(i % (test_data.Count/100) == 0)
			{
				Console.Write(".");
				GC.Collect();
			}
		}
		Console.WriteLine("Writing submission file...");
		submission_file = new StreamWriter(submission_filename + string.Format("{0:yyMMddHHmmss}", dt));
		submission_file.WriteLine("id,target");
		for(int i = 0; i < predictions.Count; i++)
			submission_file.WriteLine(i + "," + predictions[i].ToString("F6"));
		submission_file.Close();
	}


	private void SetupRecommender(string parameters)
	{
		recommender.Configure(parameters);
	}


}
