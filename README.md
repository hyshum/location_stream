# Routine Home & Work

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
Written in Python 2.7. Numpys, Pandas, and Scikit-learn needed.

## Problem Statement
Given a stream of location history determine the location of a persons home and work.  
Read data from a csv file with the following columns: (posix time, utc offset, latitude, longitude, motion state).
Motion state is a enum with these possible values:  <br />
Stationary = 0 <br />
Walking = 1 <br />
Running = 2 <br />
Vehicular = 3 <br />

Attached is an naive & ideal example of someones location history. How would you generate your own history?

## Determine home and work
To determine the location of a persons home and work, I assume that the place you are stationary most frequently is home
and the place you are stationary 2nd most frequently is work. I filtered out the non-stationary rows of data (motion_state != 0)

## Result
Home coords: (37.767588, -122.496692) Golden Gate Park <br />
Work coords: (37.331996999999994, -122.02961100000002) 1 Infinite Loop 

## Improvement
The raw data is clean in that the latitude and longtitude measures taken at home and at work are precisely the same. This is not a good representation of the real world full of interferences; it also would not capature small movements within home or work. Therefore, a gaussian noise is added to the stationary data to account for the statistical error. Data are fuzzed according to a given standard deviation.

## Algorithms
In the process of developing an algorithm for this problem, multiple algorithm has been considered and experienced with.
1. Create windows of location coordinates so that fuzzed data can be capture and the mean could be the location. This algorithm fails when there is high fuzzing and locations just exceeds the limits of the windows. <br />
2. Detect entering/leaving home and work can also be challenging because there could also be motion state error during the event.<br />
3. K-Means Clustering can solve the problem of statistical noise being too big. Experimenting this algorithm with the stationary data, I find that an increase in number of clusters can lead to more accurate predictions. Yet, this method is too sensitive of outlier that can be found in fuzzed data.<br />
4. Mean-shift Clustering is chosen as the final algorithm for this problem because it is more robust to outliers. With the stationary data, the highest mode is home , and the second highest mode is work.

## Benchmark
The benchmark for the predictions is their distance from the groud truth: Golden Gate Park and 1 Infinite Loop.

## Final Results
### Very Low Fuzz , Standard deviation : 0.5 meters 
Predicted Home coords: [  37.76758799 -122.49669198] (error: 0.00204839260291 meters) <br />
Predicted Work coords: [  37.33199716 -122.02961107] (error: 0.0191743861948 meters) <br />

### Low Fuzz, Standard deviation: 3 meters
Predicted Home coords: [  37.76758797 -122.4966914 ] (error: 0.0672033832141 meters) <br />
Predicted Work coords: [  37.33199608 -122.02960987] (error: 0.156302615043 meters) <br />

### Moderate Fuzz, Standard deviation: 10 meters
Predicted Home coords: [  37.76759057 -122.49669073] (error: 0.318652627125 meters) <br />
Predicted Work coords: [  37.33199108 -122.02961107] (error: 0.658203211494 meters) <br />

### High Fuzz, Standard deviation: 1 kilometer
Predicted Home coords: [  37.76758605 -122.49628467] (error: 45.1877835674 meters) <br />
Predicted Work coords: [  37.33285503 -122.02925269] (error: 102.401847787 meters)<br />

### Very High Fuzz, Standard deviation: 10 kilometers
Predicted Home coords: [  37.76367262 -122.5006027 ] (error: 614.659563725 meters)<br />
Predicted Work coords: [  37.34009755 -122.03065612] (error: 907.26764553 meters)<br />

## Conclusion
Errors of predictions are acceptable for all level of fuzzing. From very low to moderate, the results are very precise. At high, the result is accept for navigation on Maps. Even at very high level fuzzing, the result is reasonably accurate.<br />

For future improvement, statisical error can also be introduced to the motion state as there too could be errors. It is would be an interesting investigation to see how well this model performs with fuzzed motion state data.






  





