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
using MathNet.Numerics.LinearAlgebra.Double;
using MyMediaLite.DataType;
using MyMediaLite.Data;

namespace MyMediaLite.ItemRecommendation
{
	public class ItemAttributeCMF : MF, IItemAttributeAwareRecommender
	{

		protected Matrix<float> item_attribute_factors;

		public float Alpha { get { return alpha; } set { alpha = value; } }
		float alpha = 0.5f;

		/// <summary>Regularization parameter</summary>
		public double Regularization { get { return regularization; } set { regularization = value; } }
		double regularization = 0.032;

		/// <summary>Learn rate (update step size)</summary>
		public float LearnRate { get { return learn_rate; } set { learn_rate = value; } }
		float learn_rate = 0.31f;

		/// <summary>Multiplicative learn rate decay</summary>
		/// <remarks>Applied after each epoch (= pass over the whole dataset)</remarks>
		public float Decay { get { return decay; } set { decay = value; } }
		float decay = 1.0f;

		/// <summary>The learn rate used for the current epoch</summary>
		protected internal float current_learnrate;

		///
		public bool UpdateAttributes { get; set; }

		///
		public IBooleanMatrix ItemAttributes
		{
			get { return this.item_attributes; }
			set {
				this.item_attributes = value;
				this.NumItemAttributes = item_attributes.NumberOfColumns;
				this.MaxItemID = Math.Max(MaxItemID, item_attributes.NumberOfRows - 1);
			}
		}
		private IBooleanMatrix item_attributes;

		///
		public int NumItemAttributes { get; private set; }

		public ItemAttributeCMF ()
		{
			UpdateUsers = true;
			UpdateItems = true;
			UpdateAttributes = true;
		}

		protected override void InitModel()
		{
			base.InitModel();

			item_attribute_factors = new Matrix<float>(NumItemAttributes + 1, NumFactors);
			item_attribute_factors.InitNormal(InitMean, InitStdDev);

			current_learnrate = LearnRate;

		}

		///
		public override void Iterate()
		{
			Iterate(UpdateUsers, UpdateItems, UpdateAttributes);
		}

		public override float ComputeObjective()
		{
			return -1;
		}

		/// <summary>Iterate once over feedback data and adjust corresponding factors (stochastic gradient descent)</summary>
		/// <param name="rating_indices">a list of indices pointing to the feedback to iterate over</param>
		/// <param name="update_user">true if user factors to be updated</param>
		/// <param name="update_item">true if item factors to be updated</param>
		protected virtual void Iterate(bool update_user, bool update_item, bool update_item_attributes)
		{
			//TODO: respeitar sequencia original
			// ou não: o objetivo é ter uma sequência diferente por cada iteração.
			// só se aplica ao treino inicial
			var indexes = Feedback.RandomIndex;
			foreach (int index in indexes)
			{
				int u = Feedback.Users[index];
				int i = Feedback.Items[index];
				//Console.WriteLine("User " + u + " Item " + i);

				UpdateFactors(u, i, update_user, update_item, update_item_attributes);
			}

			UpdateLearnRate();

		}

		///
		protected float Predict(int user_id, int item_id, bool bound)
		{
			if (user_id >= user_factors.dim1 || item_id >= item_factors.dim1)
				return float.MinValue;

			float result = DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

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
		protected float PredictItemAttribute(int user_id, int item_attribute_id, bool bound)
		{
			if (user_id >= user_factors.dim1 || item_attribute_id >= item_attribute_factors.dim1)
				return float.MinValue;

			float result = DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_attribute_factors, item_attribute_id);

			if (bound)
			{
				if (result > 1)
					return 1;
				if (result < 0)
					return 0;
			}
			return result;
		}

		/// <summary>Updates <see cref="current_learnrate"/> after each epoch</summary>
		protected virtual void UpdateLearnRate()
		{
			current_learnrate *= Decay;
		}


		///
		public override void AddFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			Retrain(feedback);
		}

		///
		public override void RemoveFeedback(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			Retrain(feedback);
		}

		void Retrain(System.Collections.Generic.ICollection<Tuple<int, int>> feedback)
		{
			foreach (var entry in feedback)
				UpdateFactors(entry.Item1, entry.Item2, UpdateUsers, UpdateItems, UpdateAttributes);
		}

		///
		protected override void AddUser(int user_id)
		{
			base.AddUser(user_id);

			user_factors.AddRows(user_id + 1);
			user_factors.RowInitNormal(user_id, InitMean, InitStdDev);
		}

		///
		protected override void AddItem(int item_id)
		{
			base.AddItem(item_id);

			item_factors.AddRows(item_id + 1);
			item_factors.RowInitNormal(item_id, InitMean, InitStdDev);
		}


		///
		public override void RemoveUser(int user_id)
		{
			base.RemoveUser(user_id);

			// set user latent factors to zero
			user_factors.SetRowToOneValue(user_id, 0);
		}

		///
		public override void RemoveItem(int item_id)
		{
			base.RemoveItem(item_id);

			// set item latent factors to zero
			item_factors.SetRowToOneValue(item_id, 0);

		}

		/// <summary>
		/// Performs factor updates for a user and item pair.
		/// </summary>
		/// <param name="user_id">User_id.</param>
		/// <param name="item_id">Item_id.</param>
		protected virtual void UpdateFactors(int user_id, int item_id, bool update_user, bool update_item, bool update_item_attributes)
		{
			//Console.WriteLine(float.MinValue);
			float err = 1 - Predict(user_id, item_id, false);
			float attr_err = 0f;
			int item_attribute_id = -1;
			bool item_has_attribute = item_attributes.GetEntriesByRow(item_id).Count > 0;
			if (item_has_attribute)
			{
				item_attribute_id = item_attributes.GetEntriesByRow(item_id)[0];
				attr_err = 1 - PredictItemAttribute(user_id, item_attribute_id, false);
			}

			// adjust factors
			for (int f = 0; f < NumFactors; f++)
			{
				float u_f = user_factors[user_id, f];
				float i_f = item_factors[item_id, f];
				float a_f = 0f;
				if(item_has_attribute)
					a_f = item_attribute_factors[item_attribute_id, f];
				
				// if necessary, compute and apply updates
				if (update_user)
				{
					double delta_u = err * i_f - Regularization * u_f;
					if(item_has_attribute)
						delta_u = (1 - Alpha) * delta_u + 
						          Alpha * (attr_err * a_f - Regularization * u_f);
					user_factors.Inc(user_id, f, current_learnrate * delta_u);
				}
				if (update_item)
				{
					double delta_i = err * u_f - Regularization * i_f;
					item_factors.Inc(item_id, f, current_learnrate * delta_i);
				}
				if(update_item_attributes && item_has_attribute)
				{
					double delta_attr = attr_err * u_f - Regularization * a_f;
					item_attribute_factors.Inc(item_attribute_id, f, current_learnrate * delta_attr);
				}
			}

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
						float score = -Math.Abs(1 - Predict(user_id, item_id));
						if (score > float.MinValue)
							scored_items.Add(Tuple.Create(item_id, score));
					}

				/*				
				Parallel.ForEach(candidate_items, item_id => {
					if (!ignore_items.Contains(item_id))
					{
						float score = -Math.Abs(1 - Predict(user_id, item_id));
						if (score > float.MinValue)
							lock(scored_items)
								scored_items.Add(Tuple.Create(item_id, score));
					}
				});
				*/
				ordered_items = scored_items.OrderByDescending(x => x.Item2).ToArray();
			}
			else
			{
				var comparer = new DelegateComparer<Tuple<int, float>>( (a, b) => a.Item2.CompareTo(b.Item2) );
				var heap = new IntervalHeap<Tuple<int, float>>(n, comparer);
				float min_relevant_score = float.MinValue;

				foreach (int item_id in candidate_items)
					if (!ignore_items.Contains(item_id))
					{
						float score = -Math.Abs(1 - Predict(user_id, item_id));
						if (score > min_relevant_score)
						{
							heap.Add(Tuple.Create(item_id, score));
							if (heap.Count > n)
							{
								heap.DeleteMin();
								min_relevant_score = heap.FindMin().Item2;
							}
						}
					}

				/*				
				Parallel.ForEach(candidate_items, item_id => {
					if (!ignore_items.Contains(item_id))
					{
						float score = -Math.Abs(1 - Predict(user_id, item_id));
						if (score > min_relevant_score)
						{
							lock (heap)
							{
								heap.Add(Tuple.Create(item_id, score));
								if (heap.Count > n)
								{
									heap.DeleteMin();
									min_relevant_score = heap.FindMin().Item2;
								}
							}
						}
					}
				});
				*/
				ordered_items = new Tuple<int, float>[heap.Count];
				for (int i = 0; i < ordered_items.Count; i++)
					ordered_items[i] = heap.DeleteMax();
			}

			return ordered_items;
		}


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"ItemAttributeCMF num_factors={0} regularization={1} learn_rate={2} num_iter={3} decay={4} alpha={5}",
				NumFactors, Regularization, LearnRate, NumIter, Decay, Alpha);
		}


	}
}

