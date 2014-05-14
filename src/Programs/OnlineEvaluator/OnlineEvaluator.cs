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
	bool repeated_items = false;
	IncrementalItemRecommender recommender;
	IMapping user_mapping = new Mapping();
	IMapping item_mapping = new Mapping();
	IPosOnlyFeedback train_data;
	IPosOnlyFeedback test_data;
	IDictionary<string, IList<double>> measures;
	IBooleanMatrix user_attributes;
	IBooleanMatrix item_attributes;
	string[] metrics = new string[]{"recall@1","recall@5","recall@10","recall@20","MAP","AUC","NDCG","rec_time"};
	string[] output_recs_buffer;
	int output_recs_buffer_count = 0;
	string[] output_measure_buffer;
	int output_measure_buffer_count = 0;
	string[] output_times_buffer;
	int output_times_buffer_count = 0;
	StreamWriter output_recs;
	StreamWriter output_scores;
	StreamWriter output_times;
	int output_interval = 1000;

	public OnlineEvaluator(string[] args)
	{
		if(args.Length < 4) {
			Console.WriteLine("Usage: online_evaluator <recommender> <\"recommender params\"> <training_file> <test_file> [<random_seed> [<n_recs> [<repeated_items> [<output_interval> [<attributes>]]]]]");
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

		if(args.Length > 6) repeated_items = Boolean.Parse(args[6]);

		if(args.Length > 7) output_interval = Int32.Parse(args[7]);

		if(args.Length > 8)
		{
			if(recommender is IUserAttributeAwareRecommender) 
			{
				user_attributes = AttributeData.Read(args[8], user_mapping);
				((IUserAttributeAwareRecommender)recommender).UserAttributes = user_attributes;
			}
			else if(recommender is IItemAttributeAwareRecommender)
			{
				item_attributes = AttributeData.Read(args[8], item_mapping);
				((IItemAttributeAwareRecommender)recommender).ItemAttributes = item_attributes;
			}
		}

		measures = InitMeasures();

		DateTime dt = DateTime.Now;

		output_scores = new StreamWriter("ov" + method + args[3].Substring(args[3].LastIndexOf("/")+1) + String.Format("{0:yyMMddHHmmss}", dt) + ".log");
		output_scores.WriteLine("idx\tuser\titem\trecall@1\trecall@5\trecall@10\trecall@20\tmap\tauc\tndcg\trec_time");
		output_times = new StreamWriter("update_times" + method + args[3].Substring(args[3].LastIndexOf("/")+1) + String.Format("{0:yyMMddHHmmss}", dt) + ".log");
		output_times.WriteLine("idx\tuser\titem\tretrain_time");
		output_recs = new StreamWriter("recs" + method + args[3].Substring(args[3].LastIndexOf("/")+1) + String.Format("{0:yyMMddHHmmss}", dt) + ".log");

		output_recs_buffer = new string[output_interval];
		output_measure_buffer = new string[output_interval];
		output_times_buffer = new string[output_interval];

	}

	public static void Main(string[] args)
	{
		var program = new OnlineEvaluator(args);
		program.Run();
	}
	
	private void Run()
	{
		double recall1,recall5,recall10,recall20,map,ndcg,auc,rec_time;
		int tu,ti,num_dropped;
		bool recommend;
		HashSet<int> ignore_items;
		IList<Tuple<int,float>> rec_list_score;
		List<int> rec_list,til;
		Tuple<int,int> tuple;
		var candidate_items = new List<int>(train_data.AllItems.Union(test_data.AllItems));

		recommender.Feedback = train_data;

		output_recs.WriteLine(recommender.ToString());

		DateTime start_train = DateTime.Now;
		recommender.Train();
		TimeSpan train_time = DateTime.Now - start_train;

		output_recs.WriteLine("Train time: " + train_time.TotalMilliseconds);

		DateTime rec_start, rec_end, retrain_start, retrain_end;

		for (int i = 0; i < test_data.Count; i++)
		{
			tu = test_data.Users[i];
			ti = test_data.Items[i];
			//Console.WriteLine("\n" + tu + " " + ti);
			output_recs_buffer[output_recs_buffer_count] = "\n" + tu + " " + ti + "\n";
			if (train_data.AllUsers.Contains(tu))
			{
				recommend = true;
				if(!repeated_items)
					if(train_data.UserMatrix[tu].Contains(ti))
						recommend = false;
				if(recommend)
				{
					if(repeated_items)
						ignore_items = new HashSet<int>();
					else
						ignore_items = new HashSet<int>(train_data.UserMatrix[tu]);
					rec_start = DateTime.Now;
					rec_list_score = recommender.Recommend(tu, n_recs, ignore_items, candidate_items);
					rec_end = DateTime.Now;
					rec_list = new List<int>();
					foreach (var rec in rec_list_score)
					{
						output_recs_buffer[output_recs_buffer_count] += rec.Item1 + "\t" + rec.Item2 + "\n";
						rec_list.Add(rec.Item1);
					}
					output_recs_buffer_count++;

					til = new List<int>(){ ti };

					recall1 = MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 1);
					recall5 = MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 5);
					recall10 = MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 10);
					recall20 = MyMediaLite.Eval.Measures.PrecisionAndRecall.RecallAt(rec_list, til, 20);
					map = MyMediaLite.Eval.Measures.PrecisionAndRecall.AP(rec_list, til);
					num_dropped = candidate_items.Count - ignore_items.Count - n_recs;
					auc = MyMediaLite.Eval.Measures.AUC.Compute(rec_list, til, num_dropped);
					ndcg = MyMediaLite.Eval.Measures.NDCG.Compute(rec_list, til);
					rec_time = (rec_end - rec_start).TotalMilliseconds;
					measures["recall@1"].Add(recall1);
					measures["recall@5"].Add(recall5);
					measures["recall@10"].Add(recall10);
					measures["recall@20"].Add(recall20);
					measures["MAP"].Add(map);
					measures["AUC"].Add(auc);
					measures["NDCG"].Add(ndcg);
					measures["rec_time"].Add(rec_time);
					output_measure_buffer[output_measure_buffer_count++] = i + "\t" + user_mapping.ToOriginalID(tu) + "\t" + 
						item_mapping.ToOriginalID(ti) + "\t" + recall1 + "\t" +
						recall5 + "\t" + recall10 + "\t" + recall20 + "\t" + map + "\t" +
						auc + "\t" + ndcg + "\t" + rec_time;
				}
			}
			// update recommender
			tuple = Tuple.Create(tu, ti);
			retrain_start = DateTime.Now;
			recommender.AddFeedback(new Tuple<int, int>[]{ tuple });
			retrain_end = DateTime.Now;
			measures["retrain_times"].Add((retrain_end - retrain_start).TotalMilliseconds);
			output_times_buffer[output_times_buffer_count++] = i + "\t" + user_mapping.ToOriginalID(tu) + "\t" + 
				item_mapping.ToOriginalID(ti) + "\t" + (retrain_end - retrain_start).TotalMilliseconds;
			WriteOutputBuffer();
			if(i % 5000 == 0)
				System.GC.Collect();
		}

		Terminate();

	}

	private void WriteOutputBuffer(bool final = false) {
		if (output_times_buffer_count == output_interval || final)
		{
			for (int i = 0; i < output_times_buffer_count; i++)
				output_times.WriteLine(output_times_buffer[i]);
			output_times_buffer_count = 0;
		}
		if (output_recs_buffer_count == output_interval || final)
		{
			for (int i = 0; i < output_recs_buffer_count; i++)
				output_recs.Write(output_recs_buffer[i]);
			output_recs_buffer_count = 0;
		}
		if (output_measure_buffer_count == output_interval || final)
		{
			for (int i = 0; i < output_measure_buffer_count; i++)
				output_scores.WriteLine(output_measure_buffer[i]);
			output_measure_buffer_count = 0;
		}
	}

	private void SetupRecommender(string parameters)
	{
		recommender.Configure(parameters);
	}

	private IDictionary<string, IList<double>> InitMeasures()
	{
		var dict = new Dictionary<string, IList<double>>();
		foreach (var metric in metrics)
			dict.Add(metric, new List<double>());
		dict.Add("retrain_times", new List<double>());
		return dict;
	}

	private void Terminate()
	{

		WriteOutputBuffer(true);
		//Compute and print averages
		output_recs.WriteLine();
		foreach (var measure in measures)
		{
			double score = Math.Round(measure.Value.Average(), 5);
			output_recs.WriteLine(measure.Key + ":\t" + score);
	
		}

		output_scores.Close();
		output_times.Close();
		output_recs.Close();
	}

}
