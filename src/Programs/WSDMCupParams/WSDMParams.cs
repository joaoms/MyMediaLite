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
using MyMediaLite.ItemRecommendation;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Linq;

class WSDMCupParams
{
	int random_seed = 10;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	Dictionary<int,string[]> item_features;
	Tuple<PosOnlyFeedback<SparseBooleanMatrix>,List<int>> all_data, train_data, test_data;
	DateTime dt = DateTime.Now;
	double split;
	string parameters, test_param;
	string[] test_param_values;
	int max_cores = 16;
	readonly char[] SPLITCHARS = { '\t', ',' };


	public WSDMCupParams(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: wsdm_cup_task2 <\"recommender params\"> <test_param> <\"test_param_values\"> <data_file> <split>[ <random_seed>[ <item_feature_file> <\"item_features\">[ <max_cores>]]]");
			Environment.Exit(1);
		}

		parameters = args[0];

		test_param = args[1];
		test_param_values = Regex.Split(args[2], "\\s");

		Console.WriteLine("Reading data...");
		all_data = ReadTrainData(args[3]);

		split = double.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture);

		SplitData();

		if(args.Length > 5) random_seed = int.Parse(args[5]);
		MyMediaLite.Random.Seed = random_seed;

		if(args.Length > 6)
		{
			Console.WriteLine("Reading item features...");
			item_features = ReadItemData(args[6], args[7]);
			train_data = InsertVirtualItems();
		}

		if(args.Length > 8) max_cores = int.Parse(args[8]);

	}

	public static void Main(string[] args)
	{
		var program = new WSDMCupParams(args);
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

	private Dictionary<int,string[]> ReadItemData(string filename, string cols)
	{
		var ret = new Dictionary<int, string[]>();
		var reader = new StreamReader(filename);
		var colnum_str = cols.Split(',');
		var colnums = new int[colnum_str.Length];
		int colmax = 0;
		for (int i = 0; i < colnum_str.Length; i++)
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
				var attr = new string[colnum_str.Length];
				for (int i = 0; i < attr.Length; i++)
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

	private void SplitData()
	{
		if(split <= 0 || split >=1) 
		{
			Console.WriteLine("Split value must be between 0 and 1 (excl)!");
			Environment.Exit(1);
		}

		train_data = Tuple.Create(new PosOnlyFeedback<SparseBooleanMatrix>(), new List<int>());
		test_data = Tuple.Create(new PosOnlyFeedback<SparseBooleanMatrix>(), new List<int>());

		int cnt = all_data.Item2.Count;
		int split_idx = (int) Math.Ceiling(cnt * split);
		//Console.WriteLine("Split point and index " + split_idx);
		for (int i = 0; i < cnt; i++)
		{
			if(i < split_idx) 
			{
				train_data.Item1.Add(all_data.Item1.Users[i], all_data.Item1.Items[i]);
				train_data.Item2.Add(all_data.Item2[i]);
			} else {
				test_data.Item1.Add(all_data.Item1.Users[i], all_data.Item1.Items[i]);
				test_data.Item2.Add(all_data.Item2[i]);
			}
		}
	}

	public double ComputeAUC(IList<double> probability, IList<int> truth)
	{


		double auc = 0;
		int nfalse = 0;

		if (truth.Count != probability.Count) {
			throw new Exception("The vector sizes don't match");
		}

		var count = truth.Count;
		var pairs = new List<Tuple<double, int>>(count);

		for (int i = 0; i < count; i++)
			pairs.Add(Tuple.Create(probability[i], truth[i]));

		pairs.OrderBy(x => x.Item1);


		for (int i = 0; i < count; i++)
		{
			nfalse += (1 - pairs[i].Item2);
			auc += (pairs[i].Item2 * nfalse);
		}

		return (auc / (nfalse * (count - nfalse)));
	}

	private void Run()
	{
		Console.WriteLine(test_param + "\tAUC");
		var parallel_opts = new ParallelOptions() { MaxDegreeOfParallelism = max_cores };
		Parallel.ForEach(test_param_values, parallel_opts, param_val => {

			ISGDWSDM recommender = new ISGDWSDM();

			recommender.Configure(parameters);
			recommender.SetProperty(test_param, param_val);

			recommender.Feedback = train_data.Item1;
			recommender.scores = train_data.Item2;

			var candidate_items = recommender.Feedback.AllItems;
			var predictions = new List<double>(test_data.Item2.Count);

			recommender.Train();

			for (int i = 0; i < test_data.Item2.Count; i++)
			{
				int tu = test_data.Item1.Users[i];
				int ti = test_data.Item1.Items[i];
				predictions.Add(Math.Max(Math.Min(recommender.Predict(tu, ti), 1d), 0d));

				if(i % (test_data.Item2.Count/100) == 0)
				{
					GC.Collect();
				}
			}

			var auc = ComputeAUC(predictions, test_data.Item2);
			Console.WriteLine(param_val + "\t" + auc);
		});
	}

}
