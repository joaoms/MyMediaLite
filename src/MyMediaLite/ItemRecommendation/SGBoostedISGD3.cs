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
using C5;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MyMediaLite.Data;
using MyMediaLite.DataType;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>
	///   Boosting with Incremental Stochastic Gradient Descent (BoostedISGD) algorithm for item prediction. 
	/// 	This version uses online AdaBoost based on recall@N (hit or miss) obtained by each weak model.
	/// </summary>
	/// <remarks>
	///   <para>
	///     Literature:
	/// 	<list type="bullet">
	///       <item><description>
	///         João Vinagre, Alípio Mário Jorge, João Gama:
	///         
	///       </description></item>
	///     </list>
	///   </para>
	///   <para>
	///       This algorithm is primarily designed to use with incremental learning, 
	/// 		batch behavior has not been thoroughly studied.
	///   </para> 
	///   <para>
	///     This algorithm supports (and encourages) incremental updates. 
	///   </para>
	/// </remarks>
	public class SGBoostedISGD3 : IncrementalItemRecommender, IIterativeModel
	{
		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.032;

		/// <summary>Learn rate (update step size)</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		float learn_rate = 0.31f;

		/// <summary>Learn rate (update step size)</summary>
		public float BoostingLearnRate { get { return boosting_learn_rate; } set { boosting_learn_rate = value; } }
		float boosting_learn_rate = 0.31f;

		/// <summary>Multiplicative learn rate decay</summary>
		/// <remarks>Applied after each epoch (= pass over the whole dataset)</remarks>
		public float Decay { get { return decay; } set { decay = value; } }
		float decay = 1.0f;

		/// <summary>Number of latent factors per user/item</summary>
		public uint NumFactors { get { return (uint) num_factors; } set { num_factors = (int) value; } }
		/// <summary>Number of latent factors per user/item</summary>
		protected int num_factors = 10;


		/// <summary>Number of iterations over the training data</summary>
		public uint NumIter { get { return num_iter; } set { num_iter = value; } }
		uint num_iter = 10;

		/// <summary>Incremental iteration number</summary>
		public uint IncrIter { get; set; }

		/// <summary>Incremental iteration number</summary>
		public bool UseMulticore { get { return use_multicore; } set { use_multicore = value; } }
		bool use_multicore = true;

		/// <summary>Number of bootstrap nodes.</summary>
		public int NumNodes { get { return num_nodes; } set { num_nodes = value; } }
		int num_nodes = 4;

		///
		protected MyMediaLite.Random rand;
		
		/// 
		protected List<ISGD> recommender_nodes;

		///
		protected double[] predictions, errors;

		///
		public SGBoostedISGD3 ()
		{
			UpdateUsers = true;
			UpdateItems = true;
			rand = MyMediaLite.Random.GetInstance();
		}

		///
		protected virtual void InitModel()
		{
			recommender_nodes = new List<ISGD>(num_nodes);
			predictions = new double[num_nodes];
			errors = new double[num_nodes];
			ISGD recommender_node;
			IPosOnlyFeedback train_data;
			for (int i = 0; i < num_nodes; i++) {
				recommender_node = new ISGD();
				recommender_node.UpdateUsers = true;
				recommender_node.UpdateItems = true;
				recommender_node.Regularization = Regularization;
				recommender_node.NumFactors = NumFactors;
				recommender_node.LearnRate = LearnRate;
				recommender_node.IncrIter = IncrIter;
				recommender_node.NumIter = NumIter;
				recommender_node.Decay = Decay;
				train_data = new PosOnlyFeedback<SparseBooleanMatrix>();
				for (int j = 0; j < Feedback.Count; j++)
					train_data.Add(Feedback.Users[j], Feedback.Items[j]);
				recommender_node.Feedback = train_data;
				recommender_nodes.Add(recommender_node);
			}
		}

		///
		public override void Train()
		{
			InitModel();
			Parallel.ForEach(recommender_nodes, rnode => { 
				rnode.Train(); 
			});
		}

		///
		public virtual void Iterate()
		{
			Parallel.ForEach(recommender_nodes, rnode => { rnode.Iterate(); });
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, false);
		}

		///
		protected virtual float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id > recommender_nodes[0].MaxUserID || item_id >= recommender_nodes[0].MaxItemID)
				return float.MinValue;

			float result = 0;
			float p = 0;
			int n = 0;

			for (int i = 0; i < num_nodes; i++)
			{
				if (!float.IsNaN(p = recommender_nodes[i].Predict(user_id, item_id)))
				{
					result += p;
					n++;
				}
			}

			if (n > 0)
				result /= n;

			if (bound)
			{
				if (result > 1)
					return 1;
				if (result < 0)
					return 0;
			}
			return result;
		}

		///
		public override void AddFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			int user, item;
			base.AddFeedback(feedback);
			foreach (var entry in feedback)
			{
				user = entry.Item1;
				item = entry.Item2;

				double gradient = 0;
				double residual = 0;
				double target = 1;

				for (int i = 0; i < num_nodes; i++)
				{
					recommender_nodes[i].AddFeedbackRetrainN(new Tuple<int,int>[] {entry}, 0);
					predictions[i] = recommender_nodes[i].Predict(user, item);
					errors[i] = Math.Max(Math.Min(target - (boosting_learn_rate * predictions[i]), 1), 0);
					target = errors[i];
				}
				recommender_nodes[0].Retrain(new Tuple<int,int>[] {entry}, 1);
				for (int i = 1; i < num_nodes; i++)
				{
					gradient = 2 * boosting_learn_rate * errors[i-1];
					recommender_nodes[i].Retrain(new Tuple<int,int>[] {entry}, residual	+ gradient);
					residual += gradient - predictions[i];
				}
			}
		}

	
		///
		public override void RemoveFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			foreach (var rnode in recommender_nodes)
				rnode.RemoveFeedback(feedback);
		}

		///
		public override System.Collections.Generic.IList<Tuple<int, float>> Recommend(
			int user_id, int n = -1,
			System.Collections.Generic.ICollection<int> ignore_items = null,
			System.Collections.Generic.ICollection<int> candidate_items = null)
		{
			if (candidate_items == null)
				candidate_items = Enumerable.Range(0, MaxItemID - 1).ToList();
			if (ignore_items == null)
				ignore_items = new int[0];

			System.Collections.Generic.IList<Tuple<int, float>> ordered_items;

			if (n == -1)
			{
				var scored_items = new List<Tuple<int, float>>();
				foreach (int item_id in candidate_items)
					if (!ignore_items.Contains(item_id))
				{
					float error = Math.Abs(1 - Predict(user_id, item_id));
					if (error > float.MaxValue)
						error = float.MaxValue;
					scored_items.Add(Tuple.Create(item_id, error));
				}

				ordered_items = scored_items.OrderBy(x => x.Item2).ToArray();
			}
			else
			{
				var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
				var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
				float max_error = float.MaxValue;

				foreach (int item_id in candidate_items)
					if (!ignore_items.Contains(item_id))
				{
					float error = Math.Abs(1 - Predict(user_id, item_id));
					if (error < max_error)
					{
						heap.Add(Tuple.Create(item_id, error));
						if (heap.Count > n)
						{
							heap.DeleteMax();
							max_error = heap.FindMax().Item2;
						}
					}
				}

				ordered_items = new Tuple<int, float>[heap.Count];
				for (int i = 0; i < ordered_items.Count; i++)
					ordered_items[i] = heap.DeleteMin();
			}

			return ordered_items;
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"SGBoostedISGD num_factors={0} regularization={1} learn_rate={2} num_iter={3} incr_iter={4} decay={5} num_nodes={6} boosting_learn_rate={7}",
				NumFactors, Regularization, LearnRate, NumIter, IncrIter, Decay, NumNodes, BoostingLearnRate);
		}


	}
}

