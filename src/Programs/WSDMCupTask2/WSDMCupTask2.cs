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
using MyMediaLite.DataType;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;

class WSDMCupTask2
{
	string method = "ISGDWSDM";
	int random_seed = 10;
	ISGDWSDM recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	//Dictionary<int,List<string>> user_features;
	Dictionary<int,string[]> item_features;
	Tuple<PosOnlyFeedback<SparseBooleanMatrix>,List<int>> train_data;
	List<Tuple<int,int>> test_data;
	string submission_filename;
	StreamWriter submission_file, log_file;
	DateTime dt = DateTime.Now;
	readonly char[] SPLITCHARS = new char[] { '\t', ',' };


	public WSDMCupTask2(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: wsdm_cup_task2 <\"recommender params\"> <train_file> <test_file> <submission_file>[ <random_seed>[ <item_feature_file> <\"item_features\">]]");
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
		SetupRecommender(args[0]);
		log_file.WriteLine(recommender.ToString());

		Console.WriteLine("Reading train data...");
		train_data = ReadTrainData(args[1]);

		Console.WriteLine("Reading test data...");
		test_data = ReadTestData(args[2]);

		submission_filename = args[3];

		if(args.Length > 4) random_seed = int.Parse(args[4]);
		MyMediaLite.Random.Seed = random_seed;

		if(args.Length > 5)
		{
			Console.WriteLine("Reading item features...");
			item_features = ReadItemData(args[5], args[6]);
		}

		Console.WriteLine("Adding item features as virtual items...");
		train_data = InsertVirtualItems();


	}

	public static void Main(string[] args)
	{
		var program = new WSDMCupTask2(args);
		program.Run();
	}

	private Tuple<PosOnlyFeedback<SparseBooleanMatrix>,List<int>> ReadTrainData(string filename)
	{
		var ret_ui = new PosOnlyFeedback<SparseBooleanMatrix>();
		var ret_r = new List<int>();
		var reader = new StreamReader(filename);

		string line = reader.ReadLine();
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(SPLITCHARS);

			if (tokens.Length < 6)
				throw new FormatException("Expected at least 6 columns: " + line);

			try
			{
				int user_id = user_mapping.ToInternalID(tokens[0]);
				int item_id = item_mapping.ToInternalID(tokens[1]);
				int rating = int.Parse(tokens[5]);
				ret_ui.Add(user_id, item_id);
				ret_r.Add(rating);
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}

		return Tuple.Create(ret_ui, ret_r);

	}

	private List<Tuple<int,int>> ReadTestData(string filename)
	{
		var ret = new List<Tuple<int, int>>();
		var reader = new StreamReader(filename);

		string line = reader.ReadLine();
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(SPLITCHARS);

			if (tokens.Length < 2)
				throw new FormatException("Expected at least 2 columns: " + line);

			try
			{
				int user_id = user_mapping.ToInternalID(tokens[1]);
				int item_id = item_mapping.ToInternalID(tokens[2]);
				ret.Add(Tuple.Create(user_id, item_id));
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}
		return ret;

	}

	private Dictionary<int,string[]> ReadItemData(string filename, string cols)
	{
		var ret = new Dictionary<int, string[]>();
		var reader = new StreamReader(filename);
		var colnum_str = cols.Split(',');
		var colnums = new int[cols.Length];
		int colmax = 0;
		for (int i = 0; i < cols.Length; i++)
		{
			colnums[i] = int.Parse(colnum_str[i]);
			if (colnums[i] > colmax) colmax = colnums[i];
		}

		string line = reader.ReadLine();
		while ((line = reader.ReadLine()) != null)
		{
			if (line.Trim().Length == 0)
				continue;

			string[] tokens = line.Split(SPLITCHARS);

			if (tokens.Length < colmax)
				throw new FormatException("Expected at least " + colmax + " columns: " + line);

			try
			{
				int item_id = item_mapping.ToInternalID(tokens[0]);
				var attr = new string[cols.Length];
				for (int i = 0; i < cols.Length; i++)
				{
					attr[i] = tokens[colnums[i]];
				}
				ret.Add(item_id, attr);
			}
			catch (Exception)
			{
				throw new FormatException(string.Format("Could not read line '{0}'", line));
			}
		}
		return ret;

	}

	private Tuple<PosOnlyFeedback<SparseBooleanMatrix>,List<int>> InsertVirtualItems()
	{
		Tuple<PosOnlyFeedback<SparseBooleanMatrix>, List<int>> new_train_data = Tuple.Create(new PosOnlyFeedback<SparseBooleanMatrix>(), new List<int>());
		for (int i = 0; i < train_data.Item1.Count; i++)
		{
			int user = train_data.Item1.Users[i];
			int item = train_data.Item1.Items[i];
			int score = train_data.Item2[i];
			new_train_data.Item1.Add(user, item);
			new_train_data.Item2.Add(score);
			if(item_features.ContainsKey(item))
			{
				var item_attrs = item_features[item];
				for (int j = 0; j < item_attrs.Length; j++)
				{
					new_train_data.Item1.Add(user, item_mapping.ToInternalID(item_attrs[j]));
					new_train_data.Item2.Add(score);
				}
			}
		}
		return new_train_data;
	}

	private void Run()
	{
		var candidate_items = recommender.Feedback.AllItems;
		var predictions = new List<double>(test_data.Count);

		recommender.Feedback = train_data.Item1;
		recommender.scores = train_data.Item2;

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
