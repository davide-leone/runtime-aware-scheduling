# runtime-aware-scheduling
Job Runtime Prediction and Runtime-Aware Scheduling in HPC systems 

Usually, jobs submitted to an HPC system are scheduled without prior knowledge on the characteristics of the job. This is probably a limitation, in fact, having the possibility to perform informed scheduling decisions could improve the efficiency of the entire system. 

For this purpose, predicting the duration of a job, by analysing only information available when the job is submitted, could be of great importance. Therefore, this work consists of two parts: 
1. Job  runtime  prediction, in which the objective is to test different Machine Learning models on historical data from a real HPC system (Marconi100 from Cineca), to find the best performing model.  
2. Runtime prediction-aware scheduling, in which the objective is to integrate the runtime predictor obtained in the first part into a scheduler (Batsim) for HPC workloads.


