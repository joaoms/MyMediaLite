MyMediaLite - a free recommender system algorithm library
=========================================================

[![Build Status](https://travis-ci.org/zenogantner/MyMediaLite.svg?branch=master)](https://travis-ci.org/zenogantner/MyMediaLite)


Features
--------

 - Dozens of different recommendation methods,
    - methods can use collaborative, attribute/content, and relational data,
 - support for incremental training for most models.
 - Ready to use:
    - Includes evaluation routines for rating and item prediction;
      quality measures MAE, NAME, RMSE, CBD, AUC, MAP, precision@N, recall@N,
      NDCG, MRR; and
    - command line tools that read a simple text-based input format.
 - Compact: Core library is about 275 KB "big".
 - Portable: Written in C#, for the .NET platform;
   runs on every architecture where Mono works: Linux, Windows, Mac OS X.


Feedback and Contributions
--------------------------

We are always happy about feedback, and encourage MyMediaLite's users to
contribute code to the project.

Just fork it on [GitHub](https://github.com/zenogantner/MyMediaLite) and send pull requests!

Bugs and feature requests can be reported on our
[mailing list](https://groups.google.com/group/mymedialite) or in our
[issue tracker](https://github.com/zenogantner/MyMediaLite/issues).


Installation
------------

See doc/Installation for installation instructions.


Documentation
-------------

See doc/ and the [website](http://mymedialite.net) for more documentation.


Citing MyMediaLite
------------------

If you use MyMediaLite for your research, it would be nice to acknowledge it
in your papers by citing the following paper:

Zeno Gantner, Steffen Rendle, Christoph Freudenthaler, Lars Schmidt-Thieme:
MyMediaLite: A Free Recommender System Library. RecSys 2011

```
@inproceedings{Gantner2011MyMediaLite,
  author    = {Zeno Gantner and Steffen Rendle and Christoph Freudenthaler and Lars Schmidt-Thieme},
  title     = {{MyMediaLite}: A Free Recommender System Library},
  booktitle = {5th ACM International Conference on Recommender Systems (RecSys 2011)},
  year      = 2011,
  location  = {Chicago, USA}
}
```


[Academic publications that use or reference MyMediaLite](https://scholar.google.de/scholar?q=mymedialite)


Contributors
------------

Thanks to the following people, who provided valuable feedback, code, or other
kinds of assistance:
Roberto Abalde, Nicholas Ampazis, Thorsten Angermann, Suhrid Balakrishnan,
Alejandro Bellogín, Christian Brauch, Fu Changhong, Subramanyeshwar Cherukuri,
Simon Dooms, Lucas Drumond, Michael Ekstrand, Christoph Freudenthaler,
Zeno Gantner, Jagadeesh Gorla, Josif Grabocka, Mark Graus, Guibing Guo,
Andreas Hoffmann, Tomas Horvath, Kenneth Hoste, Frantisek Hrdina, Jia Huang,
Nicolas Hug, Dominik Imrich, Dietmar Jannach, Peng Jiang, KwangSeob Kim,
Artus Krohn-Grimberghe, Tobias Lang, Christina Lichtenthäler, Damir Logar,
Marcelo Manzato, Brian McFee, Greg Najda, Chris Newell, Thai-Nghe Nguyen,
Dimitris Paraschakis, Simon Renaud, Steffen Rendle, Marco Ribeiro,
Roland Richter, Saurabh S., Sebastian Schelter, Lars Schmidt-Thieme, Yue Shi,
Jordan Silva, Piotr Sobotka, Jessica Tölke, Tom Tung, Pieter-Jan Verbruggen,
Julien Verplanken, Elvio Vicosa, João Vinagre, Oleksandr Vitvitskyi,
Yongfeng Wang, Lina Weichbrodt, Cees Wesseling, Yong Zheng, GitHub users
jkleint and NEBabylon.

This work was funded by the European Commission FP7 project MyMedia
(Dynamic Personalization of Multimedia) under the grant agreement no. 215006.


Copyright & Licensing
---------------------

Copyright (C) 2010, 2011, 2012, 2013, 2015, 2016, 2019 The MyMediaLite contributors

MyMediaLite is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MyMediaLite is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.

----
This package contains Mono.Options, [C5](http://www.itu.dk/research/c5/), [Math.NET Numerics](http://numerics.mathdotnet.com),
and [NUnit](http://www.nunit.org).

See doc/ComponentLicenses for more information about their licensing terms.
