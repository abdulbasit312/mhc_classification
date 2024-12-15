These are the codes for the CSC 2516 Course Project

Team Members:
Mohammad Abdul Basit - abdulbasit@cs.toronto.edu
Bailey Ng
Noe Artru

Dataset is very huge (100GB), and can be shared on request, or if you have access to CS lab clusters you can find it at : "/w/247/abdulbasit/mental_health_dataset_split"

Description of files: We have kept the training files for the streams seperate so that the reviewers find it easier to run the experiments.

1.analysisScript.py -> this is used to compress the dataset to 500 tweets per user based on lexicons
2.combined.py-> this is the training and testing file for running the combined model of risky posts and symptom streams with
                projection vector. 
                Usage for training: python combined.py
                Usage for testing python combined.py --mode=test --test_ckpt=/w/331/abdulbasit/mhc_classification/ckpt_combined
