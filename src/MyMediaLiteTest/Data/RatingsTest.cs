// Copyright (C) 2010 Christina Lichtenthäler
// Copyright (C) 2011 Zeno Gantner
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
using System.Linq;
using MyMediaLite.Data;
using NUnit.Framework;

namespace MyMediaLiteTest
{
	[TestFixture()]
	public class RatingDataTest
	{
		[Test()] public void TestMaxUserIDItemID()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 7, 0.2);
			ratings.Add(6, 3, 0.3);

			Assert.AreEqual(6, ratings.MaxUserID);
			Assert.AreEqual(8, ratings.MaxItemID);
		}

		[Test()] public void TestAddRating()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 7, 0.2);
			ratings.Add(6, 3, 0.3);

			Assert.AreEqual(0.4, ratings.Get(2, 5));
			Assert.AreEqual(0.3, ratings.Get(1, 4));
			Assert.AreEqual(0.3, ratings.Get(6, 3));
		}

		[Test()] public void TestRemoveAt()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 7, 0.2);
			ratings.Add(3, 3, 0.3);
			ratings.Add(6, 3, 0.3);

			Assert.AreEqual(0.4, ratings.Get(2, 5));
			ratings.RemoveAt(ratings.GetIndex(2, 5));
			ratings.RemoveAt(ratings.GetIndex(6, 3));
			
			double r;
			Assert.IsFalse(ratings.TryGet(2, 5, out r));
		}

		[Test()] public void TestRemoveUser()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 7, 0.2);
			ratings.Add(3, 3, 0.3);

			Assert.AreEqual(0.4, ratings.Get(2, 5));
			ratings.RemoveUser(2);
			
			double rating;
			Assert.IsFalse(ratings.TryGet(2, 5, out rating));
		}

		[Test()] public void TestRemoveItem()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 4, 0.2);
			ratings.Add(3, 3, 0.3);

			Assert.AreEqual(0.2, ratings.Get(2, 4));
			ratings.RemoveItem(4);
			double r;
			Assert.IsFalse(ratings.TryGet(2, 4, out r));
		}

		[Test()] public void TestGet()
		{
			var ratings = new Ratings();
			ratings.Add(1, 4, 0.3);
			ratings.Add(1, 8, 0.2);
			ratings.Add(2, 4, 0.2);
			ratings.Add(2, 2, 0.6);
			ratings.Add(2, 5, 0.4);
			ratings.Add(3, 4, 0.2);
			ratings.Add(3, 3, 0.3);
			ratings.Add(6, 3, 0.3);

			// test Get
			Assert.AreEqual(0.2, ratings.Get(2, 4));
			Assert.AreEqual(0.3, ratings.Get(3, 3));
			Assert.AreEqual(0.3, ratings.Get(6, 3));
			
			// test index[,]
			Assert.AreEqual(0.3, ratings[1, 4]);
			Assert.AreEqual(0.2, ratings[1, 8]);
			Assert.AreEqual(0.6, ratings[2, 2]);			
		}
	}
}