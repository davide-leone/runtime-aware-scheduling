import logging
import pandas as pd
import pybatsim
from itertools import islice
from procset import ProcSet
from pybatsim.batsim.batsim import BatsimScheduler
from sklearn.tree import DecisionTreeRegressor


class RuntimePredictionAwareScheduler(BatsimScheduler):       
    # Called when the simulation starts: initializes some variables and trains the regressor using historical job data    
    def onSimulationBegins(self):
        self.nb_completed_jobs = 0

        self.jobs_completed = []
        self.jobs_waiting = []

        self.sched_delay = 0.005

        self.openJobs = set()
        self.availableResources = ProcSet((0, self.bs.nb_compute_resources - 1))
        
        self.jobs = []
        self.regressor = None
        self.train_regressor()
        
    # Loads historical job data from a CSV file and use them to train a Decision Tree Regressor model
    def train_regressor(self):
        try:
            logging.info("Loading historical data..")
            df_train = pd.read_csv("train_jobs.csv")
            logging.info("Loading complete!")

            X_train = df_train.drop(columns=['run_time'])
            y_train = df_train['run_time']

            self.regressor = DecisionTreeRegressor(random_state=13)
            logging.info("Training the regressor..")
            self.regressor.fit(X_train, y_train)
            logging.info("Training complete!")
        except Exception as e:
            logging.error(f"Error in training regressor: {e}")

    # Predicts the runtime of a job using the trained regressor
    def predict_runtime(self, job):
        try:
            job_dict = job.json_dict
            
            # Extract submission features from the job
            features = {
                'cpu': job_dict['cpu'],
                'mem (GB)': job_dict['mem'],
                'node': job_dict['node'],
                'gres/gpu': job_dict['gpu'],
                'user_id': job_dict['user_id'],
                'qos': job_dict['qos'],
                'time_limit': job_dict['time_limit']
            }

            # Convert the features into a DataFrame
            df = pd.DataFrame([features])
            # Predict the runtime using the trained model
            predicted_runtime = self.regressor.predict(df)
            
            return predicted_runtime[0]
        except Exception as e:
            logging.error(f"Error in predicting runtime: {e}")
            return float('inf')  # Return a large value to deprioritize in case of error

    # Schedules jobs based on their predicted runtimes
    def scheduleJobs(self):
        scheduledJobs = []

        print('openJobs = ', self.openJobs)
        print('available = ', self.availableResources)

        # Sort openJobs based on predicted runtimes
        sorted_openJobs = sorted(self.openJobs, key=lambda job: job.reqtime)

        # Iterating over a copy to be able to remove jobs from openJobs at traversal
        for job in sorted_openJobs:
            nb_res_req = job.requested_resources
            # Allocates resources to jobs if available
            if nb_res_req <= len(self.availableResources):
                # Retrieve the *nb_res_req* first available resources
                job_alloc = ProcSet(*islice(self.availableResources, nb_res_req))
                job.allocation = job_alloc
                scheduledJobs.append(job)
                # Updates the available resources
                self.availableResources -= job_alloc

                self.openJobs.remove(job)

        # update time
        self.bs.consume_time(self.sched_delay)

        # Executes the scheduled jobs
        if len(scheduledJobs) > 0:
            self.bs.execute_jobs(scheduledJobs)

        print('openJobs = ', self.openJobs)
        print('available = ', self.availableResources)
        print('')
    
    # Called when a new job is submitted: predicts the job's runtime and adds it to the set of open jobs
    def onJobSubmission(self, job):
        # Reject the job if it requests more resources than the machine has
        if job.requested_resources > self.bs.nb_compute_resources:
            self.bs.reject_jobs([job])
        else:
            # Predict runtime and set it for the job
            predicted_runtime = self.predict_runtime(job)
            # Set the predicted runtime as the required time
            job.reqtime = predicted_runtime  
            # Adds the job to the set of open jobs
            self.openJobs.add(job)
    
    # Called when a job is completed: releases the resources allocated to the job
    def onJobCompletion(self, job):
        self.availableResources |= job.allocation
    
    # Called when there are no more events to process: ensure any remaining jobs are scheduled
    def onNoMoreEvents(self):
        self.scheduleJobs()