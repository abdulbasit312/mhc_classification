These are the codes for the CSC 2516 Course Project

Team Members:
Mohammad Abdul Basit - abdulbasit@cs.toronto.edu
Bailey Ng - baileyng@cs.toronto.edu
Noe Artru

Dataset is very huge (~100GB), and can be shared on request, or if you have access to CS lab clusters you can find it at: "/w/247/abdulbasit/mental_health_dataset_split"

Description of files: We have kept the training files for the streams seperate so that the reviewers find it easier to run the experiments.

1. analysisScript.py -> this is used to compress the dataset to 500 tweets per user based on lexicons
2. combined.py -> this is the training and testing file for running the combined model of risky posts and symptom streams with projection vector. 
                Usage for training: python combined.py --mode=train
                Usage for testing python combined.py --mode=test --test_ckpt=/w/331/abdulbasit/mhc_classification/ckpt_combined
3. split_dataset.py -> splits dataset into train, test, and val sets for each disease and control class.
4. risky_posts_stream_experiments.py -> Used to train the risky posts stream with different modifications to weight decay, learning rate decay, and optimizers.
5. GRU_risky_posts_stream.py -> Used to train GRU iteration of risky posts stream with different learning rates.
6. symptom_stream_experiments.py -> Used to train the risky posts stream with different modifications to learning rate decay.
7. GRU_symptom_stream.py -> Used to train GRU iteration of symptom stream with different learning rates.

Repository inspired by https://github.com/chesiy/EMNLP23-PsyEx
