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
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MathNet.Numerics.Distributions;


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
	public class AdaBPRMF : IncrementalItemRecommender, IIterativeModel
	{
		/// <summary>Regularization parameter</summary>
		public float Regularization { get { return regularization; } set { regularization = value; } }
		float regularization = 0.032f;

		/// <summary>Number of latent factors per user/item</summary>
		public uint NumFactors { get { return (uint) num_factors; } set { num_factors = (int) value; } }
		/// <summary>Number of latent factors per user/item</summary>
		protected int num_factors = 10;

		/// <summary>Number of iterations over the training data</summary>
		public uint NumIter { get { return num_iter; } set { num_iter = value; } }
		uint num_iter = 10;

		/// <summary>Item bias terms</summary>
		//protected float[] item_bias;

		/// <summary>Sample positive observations with (true) or without (false) replacement</summary>
		public bool WithReplacement { get; set; }

		/// <summary>Sample uniformly from users</summary>
		public bool UniformUserSampling { get; set; } = true;

		/// <summary>Regularization parameter for the bias term</summary>
		public float BiasReg { get; set; }

		/// <summary>Learning rate alpha</summary>
		public float LearnRate { get; set; } = 0.05f;

		/// <summary>Regularization parameter for user factors</summary>
		public float RegU { get; set; } = 0.0025f;

		/// <summary>Regularization parameter for positive item factors</summary>
		public float RegI { get; set; } = 0.0025f;

		/// <summary>Regularization parameter for negative item factors</summary>
		public float RegJ { get; set; } = 0.00025f;

		/// <summary>If set (default), update factors for negative sampled items during learning</summary>
		public bool UpdateJ { get; set; } = true;

		/// <summary>Number of bootstrap nodes.</summary>
		public int NumNodes { get { return num_nodes; } set { num_nodes = value; } }
		int num_nodes = 4;

		///
		protected MyMediaLite.Random rand;

		/// 
		protected List<BPRMFforAda> recommender_nodes;

		///
		public AdaBPRMF ()
		{
			RegU = regularization;
			RegI = regularization;
			rand = MyMediaLite.Random.GetInstance();
		}

		///
		protected virtual void InitModel()
		{
			recommender_nodes = new List<BPRMFforAda>(num_nodes);
			BPRMFforAda recommender_node;
			IPosOnlyFeedback train_data;
			for (int i = 0; i < num_nodes; i++) {
				recommender_node = new BPRMFforAda();
				recommender_node.UpdateUsers = true;
				recommender_node.UpdateItems = true;
				recommender_node.RegU = RegU;
				recommender_node.RegI = RegI;
				recommender_node.RegJ = RegJ;
				recommender_node.NumFactors = NumFactors;
				recommender_node.LearnRate = LearnRate;
				recommender_node.NumIter = NumIter;
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

			int i;
			for (i = 0; i < num_nodes; i++)
				if (!float.IsNaN(p = recommender_nodes[i].Predict(user_id, item_id)))
					result += p;
			result /= i;

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
			int user, item, other_item, k;
			double lambda, err;
			base.AddFeedback(feedback);
			int[] correct = Enumerable.Repeat(0,num_nodes).ToArray();
			int[] wrong = Enumerable.Repeat(0,num_nodes).ToArray();
			foreach (var entry in feedback)
			{
				user = entry.Item1;
				item = entry.Item2;

				lambda = 1;

				SampleOtherItem(user, item, out other_item);
				for (int i = 0; i < num_nodes; i++)
				{
					k = Poisson.Sample(lambda);
					recommender_nodes[i].AddFeedbackRetrainN(new Tuple<int,int>[] {entry}, k);
					if (Predict(user, item) >= Predict(user, other_item))
					{	
						correct[i]++;
						err = wrong[i] / (correct[i] + wrong[i]);
						lambda /= 2 * ( 1 - err );
					} 
					else
					{
						wrong[i]++;
						err = wrong[i] / (correct[i] + wrong[i]);
						lambda /= 2 * err;
					}
				}

			}
		}

		 
		/// <summary>Sample another item, given the first one and the user</summary>
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the ID of the given item</param>
		/// <param name="other_item_id">the ID of the other item</param>
		/// <returns>true if the given item was already seen by user u</returns>
		protected virtual bool SampleOtherItem(int user_id, int item_id, out int other_item_id)
		{
			if (Feedback.Count >= (MaxUserID + 1) * (MaxItemID + 1)) 
			{
				other_item_id = item_id;
				return true;
			}

			bool item_is_positive = Feedback.UserMatrix[user_id, item_id];

			do
				other_item_id = rand.Next(MaxItemID + 1);
			while (Feedback.UserMatrix[user_id, other_item_id] == item_is_positive);

			return item_is_positive;
		}


		///
		public override void RemoveFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			foreach (var rnode in recommender_nodes)
				rnode.RemoveFeedback(feedback);
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"AdaBPRMF num_factors={0} regularization={1} learn_rate={2} num_iter={3} num_nodes={4}",
				NumFactors, Regularization, LearnRate, NumIter, NumNodes);
		}


	}
}

