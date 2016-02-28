import numpy as np

predictionAccuracyPerHashtag = np.array([
-2,-1, 0,-2, 0,-1,-1, 3, 2,-1,-2, 0,
-4,-3, 0, 0,-1,-2, 3,-1,-1,-1,-1, 0,
-1,  -12,-1,-2,-5,-8,-1, 1, 0, 0, 0,-2,
1,-2,-1,-1, 1,-2, 2,-1, 0, 0,-1, 1,
-2,-4,-1,35,-3, 0, 0, 0,-1,-1,-8,-3,
3,-8, 1, 1,-1,-2, 0, 2,-6,-1, 0, 0,
6,-2, 0, 0,-6,-3,14, 0,-4, 1,-1, 1,
0, 1,-3, 0,-1,-1, 0,-2, 2, 3,-1,-2,
-1, 9, 0, 0,-1, 4,-4,-4, 0, 0, 0,-6,
0, 0, 0, 0, 0, 0,-1,-3, 0,-8,-2, 0,
-1, 0, 1,-1,-2, 137, 68, 0,-1, 0,-2, 0,
-2, 0, 0,-1,-1, 0, 0, 4, 0, 0, 0, 0,
0,-2, 0,-2,-1, 0,-1, 0, 196,-2,-1, 0,
0, 0,-3, 3, 1, 0,-1, 0, 0,-1, 5,-8,
-1, 0,-1,-1,-4, 0, 0, 0,-1,-3,-1,-2,
0, 3,-2, 0,-4, -21, 0,-5, 0,-3, 2,-1,
-2, 1,-2,-8,-1, 0, 1,-2, 0,-3, 0,-1,
0, 0,-4,-4, 0,-4,-5,-1, 3,-1,-4,-2,
0, 2,-2,-3,-2,-1, 0,-1,-1,-2,-6, 0,
1, 0,-5,-2,-2,-2,-2, 0, 0,-3,-3, 0,
1, 3,-2, 1,-3, 0, 0,-1,-1,-1, 2, 0,
-3, 0, 0, 9,-1, 0, 0,-1,-1, 0,46,-1,
-2,-2,-3,-1, 0, 0,-2,-2, 2,-3,-3,-1,
1, 0,-4, 1,-1, 1, 1,-4,-2,-1,-4,-1,
0,-4, 1,-1, 0,-1,-3, 0,-3, 0, 9, 0,
0, 1, 3,-2, 0,-3, 0,-2,-3,-2, 1,-1,
-10,-1,-1,-1, 2,-1,-3, 0,-1, 2,-1, 1,
-1, 0,-4, 3,-1, 0, 0,-2,-1,-2, 0, 3,
-4, 0, 4,-1,-1, 2, 0, 0,-1,-1,-3, 0,
-4, 0,-1, 0,-5,-3, 0,-3, 0, 1, 2,-1,
7,-6,-1, 0,-1,-1,-1,-1,-3, 0, 0,-2,
-2,-1, 0,-2, 4,-3, 0,-2,-1, 0, 0,-1,
0,-1,-4, 0, 0,-4, 0,-1,-1,-4,-3,-6,
-4, 0,-1,-2, 0, 2,-6,-1, 0, 0, 0, 0,
-1,-3,-1,-1, 0,-1,-3, 0,-7,-1,-3,-4,
0, 0, 0,-2,-1,-1,-1, 1, 0,-2, 0, 0,
-4, 0, 0, 0, 0, 1, 1,-1,-2,-1,-1,-2,
0, 0,-2, 2, 0, 0,-2, 1,-2,-1,-2,-4,
0, 0, 0, 0,-1, 2, 0,-5, 9,-1,-1,-1,
1,-1, 0,-1, 0,-3,-1, 4,-4, 0, 0, 2,
3, 1,-3,-1, 0,-1,-4,-3, 2,-2,-1,-2,
0,-2, 1,-2, 0, 0, 0,-3])

indices = predictionAccuracyPerHashtag.argsort()[-20:][::-1]

for index in indices:
	print predictionAccuracyPerHashtag[index]
