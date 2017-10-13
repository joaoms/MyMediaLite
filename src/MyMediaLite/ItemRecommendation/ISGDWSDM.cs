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

namespace MyMediaLite.ItemRecommendation
{
	public class ISGDWSDM : ISGD
	{

		public List<int> scores;

		public override void Iterate ()
		{
			for (int index = 0; index < Feedback.Count; index++)
			{
				int u = Feedback.Users[index];
				int i = Feedback.Items[index];
				int r = scores[index];

				UpdateFactors(u, i, UpdateUsers, UpdateItems, r);
			}

			UpdateLearnRate();
		}

	}
}
