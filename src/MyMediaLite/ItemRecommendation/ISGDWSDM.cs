// Copyright (C) 2015 Zeno Gantner
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
using System.Linq;

namespace MyMediaLite.ItemRecommendation
{
	public class ISGDWSDM : ISGD
	{

		public List<int> scores;

		public ISGDWSDM()
		{
			InitMean = 0.5;
		}

		public override void Iterate ()
		{
			int[] indexes = Enumerable.Range(0, Feedback.Count).ToArray();
			indexes.Shuffle();
			for (int index = 0; index < Feedback.Count; index++)
			{
				int u = Feedback.Users[indexes[index]];
				int i = Feedback.Items[indexes[index]];
				int r = scores[indexes[index]];

				UpdateFactors(u, i, UpdateUsers, UpdateItems, r);
			}

			UpdateLearnRate();
		}

	}
}
