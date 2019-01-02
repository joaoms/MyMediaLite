// Copyright (C) 2011, 2012, 2013 Zeno Gantner
// Copyright (C) 2010 Zeno Gantner, Christoph Freudenthaler
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
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using MyMediaLite;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>Matrix factorization model for item prediction (ranking) optimized for BPR </summary>
	/// <remarks>
	///   <para>
	///     BPR reduces ranking to pairwise classification.
	///     The different variants (settings) of this recommender
	///     roughly optimize the area under the ROC curve (AUC).
	///   </para>
	///   <para>
	///     \f[
	///       \max_\Theta \sum_{(u,i,j) \in D_S}
	///                        \ln g(\hat{s}_{u,i,j}(\Theta)) - \lambda ||\Theta||^2 ,
	///     \f]
	///     where \f$\hat{s}_{u,i,j}(\Theta) := \hat{s}_{u,i}(\Theta) - \hat{s}_{u,j}(\Theta)\f$
	///     and \f$D_S = \{ (u, i, j) | i \in \mathcal{I}^+_u \wedge j \in \mathcal{I}^-_u \}\f$.
	///     \f$\Theta\f$ represents the parameters of the model and \f$\lambda\f$ is a regularization constant.
	///     \f$g\f$ is the  logistic function.
	///   </para>
	///   <para>
	///     In this implementation, we distinguish different regularization updates for users and positive and negative items,
	///     which means we do not have only one regularization constant. The optimization problem specified above thus is only
	///     an approximation.
	///   </para>
	///   <para>
	///     Literature:
	///     <list type="bullet">
	///       <item><description>
	///         Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme:
	///         BPR: Bayesian Personalized Ranking from Implicit Feedback.
	///         UAI 2009.
	///         http://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf
	///       </description></item>
	///     </list>
	///   </para>
	///   <para>
	///     Different sampling strategies are configurable by setting the UniformUserSampling and WithReplacement accordingly.
	///     To get the strategy from the original paper, set UniformUserSampling=false and WithReplacement=false.
	///     WithReplacement=true (default) gives you usually a slightly faster convergence, and UniformUserSampling=true (default)
	///     (approximately) optimizes the average AUC over all users.
	///   </para>
	///   <para>
	///     This recommender supports incremental updates.
	///   </para>
	/// </remarks>
	public class BPRMFforAda : BPRMF
	{

		public virtual void AddFeedbackRetrainN(System.Collections.Generic.ICollection<Tuple<int,int>> feedback, int n_retrain)
		{
			base.AddFeedback(feedback, false);
			var users = new HashSet<int>(from t in feedback select t.Item1);
			var items = new HashSet<int>(from t in feedback select t.Item2);
			for (int i = 0; i < n_retrain; i++)
			{
				if (UpdateUsers)
					foreach (int user_id in users)
						RetrainUser(user_id);
				if (UpdateItems)
					foreach (int item_id in items)
						RetrainItem(item_id);
			}
		}
	
	}
}
