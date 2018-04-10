# QuickTrain - simple training framework for Keras models

This simple framework allows for easy training of Keras models locally and on google cloud services.


# Installation and initial configuration for training on Zalando Fashion MNIST data
1. Create new project in your IDE
2. Copy content of this project into your project root directory
3. Install dependencies with pip (globally or in your virtualenv):
    * tensorflow or tensorflow-gpu (for gpu version please revert to tensorflow docs)
    * tensorboard
    * h5py
    * numpy
    * pandas (if you want test framework with fashion mnist)
    * Note: There is no need to install keras module as it is incorporated into tensorflow
4. Configure your google cloud services
    * Activate google cloud console with your google account
    * You should recieve welcome 300$ for use with validity of one year
    * Download and install Google Cloud SDK (https://cloud.google.com/sdk/)
    * In google cloud console activate following services:
        - Storage
        - ML Engine
        - Billing
        - Compute Engine
        - VPC network
    * In API configuration make sure you activate following:
        - Compute Engine API
        - Cloud Billing API
        - Cloud Machine Lerning Engine API
        - Google Cloud Storage API
    * Initialize your Google cloud SDK with following command:
        gcloud init
    * Create your bucket in google cloud storage (name must be globally unique)
        - inside create folder for your project ie. fashion
        - also create folder fashion/data for your training data
6. Configure framework
    * Downlad fashion mnist csv files from https://www.kaggle.com/zalando-research/fashionmnist/data
    * Place those files in data folder in your project
    * Run prepare_data.py to prepare data for training
    * In data subfolder you will get compiled numpy file: fashion-mnist.npz
    * Copy this file to your google storage bucket into folder data

# Framework sructure:
1. /data - folder where you keep data for processing
2. /logs - place for tensorboard logs
3. /models - place where trainer will record models
4. /trainer - trainer module
4.1 datasource.py - data source class
4.2 model.py - place where you configure your keras model
4.3 saver.py - saver class for saving trained models
4.4 trainer.py - trainer module
5. cloud-gpu.yaml - configuration for google machine learning engine
6. prepare_data.py - simple program for pre-processing before training (in our example it compile fashion-mnist csvs into numpy npz)
7. setup.py - configuration for trainer module
8. tester.py - simple example for testing trained model
9. train-gcs.bat - starts trainig on google cloud
10. train-local.bat - starts training locally

# train-gcs.bat configuration
set BUCKET=marekml-data <--- replace with your google storage backet name
set PROJECT=fashion
set DATA_FILE=fashion-mnist.npz
set CLOUD_CONFIG=cloudml-gpu.yaml
set REGION=us-east1
set RUNTIME=1.6

# train-local.bat configuration
set DATA_FILE=fashion-mnist.npz
set MODULE=trainer.trainer

# Training on GCS
For training on Google Cloud just start train-gcs.bat. As a result you will get information that job is queued.
From that moment you can see you job status and logs on google cloud console under ML Engine.
Results will be stored on your google cloud bucket under /models/job_number/
Tensorboard logs are available under /logs/job_number/

# Training locally
Start train-local.bat. I usually use this option for checking if my models are error free before I send them to cloud. But if you have fast GPU, than it is possible to train your models locally.



