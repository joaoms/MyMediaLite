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
using MyMediaLite.Correlation;
using MyMediaLite.DataType;
using MyMediaLite.IO;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>
	/// Similarity weighted M.
	/// </summary>
	public class SimilarityWeightedMF: SimpleSGD
	{
		/// <summary>Alpha parameter for BidirectionalConditionalProbability</summary>
		public float Alpha { get; set; }
		
		/// <summary>The kind of correlation to use</summary>
		public BinaryCorrelationType Correlation { get; set; }
		
		/// <summary>data matrix to learn the correlation from</summary>
		protected IBooleanMatrix DataMatrix { get { return Feedback.ItemMatrix; } }
		
		/// <summary>Correlation matrix over some kind of entity, e.g. users or items</summary>
		protected IBinaryDataCorrelationMatrix correlation;

		/// <summary>A factor for similarity weights.</summary>
		public float SimilarityWeight { get { return similarity_weight; } set { similarity_weight = value; } }
		float similarity_weight = 1.0f;

		/// 
		//protected ParallelOptions parallel_opts = new ParallelOptions() { MaxDegreeOfParallelism = 4 };
		
		public SimilarityWeightedMF ()
		{
			Correlation = BinaryCorrelationType.Cosine;
			UpdateUsers = true;
			UpdateItems = true;
			Alpha = 0.5f;
			SimilarityWeight = 1f;
		}

		/// <summary>
		/// Inits the model.
		/// </summary>
		protected override void InitModel()
		{
			base.InitModel();
			int num_entities = 0;
			switch (Correlation)
			{
			case BinaryCorrelationType.Cosine:
				correlation = new BinaryCosine(num_entities);
				break;
			case BinaryCorrelationType.Jaccard:
				correlation = new Jaccard(num_entities);
				break;
			case BinaryCorrelationType.ConditionalProbability:
				correlation = new ConditionalProbability(num_entities);
				break;
			case BinaryCorrelationType.BidirectionalConditionalProbability:
				correlation = new BidirectionalConditionalProbability(num_entities, Alpha);
				break;
			case BinaryCorrelationType.Cooccurrence:
				correlation = new Cooccurrence(num_entities);
				break;
			default:
				throw new NotImplementedException(string.Format("Support for {0} is not implemented", Correlation));
			}
		}
		
		///
		public override void Train()
		{
			base.Train();
			correlation.ComputeCorrelations(DataMatrix);
		}
		///

		public override float Predict(int user_id, int item_id)
		{
			float mf_prediction = base.Predict(user_id, item_id);
			var user_items = DataMatrix.GetEntriesByColumn(user_id);
			user_items.Remove(item_id);
			float sum = 0;
			foreach (var uitem in user_items)
				sum += correlation[item_id, uitem];
			var similarity_avg = sum / user_items.Count;

			return mf_prediction * (1 + similarity_avg * SimilarityWeight);
			 
		}


		public override void AddFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.AddFeedback(feedback);
			UpdateCorrelationMatrix(feedback);
		}
		
		///
		public override void RemoveFeedback(ICollection<Tuple<int, int>> feedback)
		{
			base.RemoveFeedback(feedback);
			UpdateCorrelationMatrix(feedback);
		}
		
		/// <summary>Update the correlation matrix for the given feedback</summary>
		/// <param name='feedback'>the feedback (user-item tuples)</param>
		protected void UpdateCorrelationMatrix(ICollection<Tuple<int, int>> feedback)
		{
			var update_entities = new HashSet<int>();
			foreach (var t in feedback)
				update_entities.Add(t.Item1);
			
			foreach (int i in update_entities)
			{
				for (int j = 0; j < correlation.NumEntities; j++)
				{
					if (j < i && correlation.IsSymmetric && update_entities.Contains(j))
						continue;
					
					correlation[i, j] = correlation.ComputeCorrelation(DataMatrix.GetEntriesByRow(i), DataMatrix.GetEntriesByRow(j));
				}
			}
		}
		

		///
		public override void SaveModel(string filename)
		{
			using ( StreamWriter writer = Model.GetWriter(filename, this.GetType(), "3.03") )
			{
				writer.WriteLine(Correlation);
				correlation.Write(writer);
			}
		}
		
		///
		public override void LoadModel(string filename)
		{
			using ( StreamReader reader = Model.GetReader(filename, this.GetType()) )
			{
				Correlation = (BinaryCorrelationType) Enum.Parse(typeof(BinaryCorrelationType), reader.ReadLine());
				
				InitModel();
				if (correlation is SymmetricCorrelationMatrix)
					((SymmetricCorrelationMatrix) correlation).ReadSymmetricCorrelationMatrix(reader);
				else if (correlation is AsymmetricCorrelationMatrix)
					((AsymmetricCorrelationMatrix) correlation).ReadAsymmetricCorrelationMatrix(reader);
				else
					throw new NotSupportedException("Unknown correlation type: " + correlation.GetType());
				
			}
		}
		
		///
		public override string ToString()
		{
			return string.Format(
				"{0} num_factors={1} num_iter={2} regularization={3} learn_rate={4} decay={5} correlation={6} similarity_weight={7}",
				this.GetType().Name, NumFactors, NumIter, Regularization, LearnRate, Decay, Correlation, SimilarityWeight);
		}
	}
}


