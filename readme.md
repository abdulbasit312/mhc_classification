These are the codes for the CSC 2516 Course Project

Team Members:
Mohammad Abdul Basit - abdulbasit@cs.toronto.edu
Bailey Ng - baileyng@cs.toronto.edu
Noe Artru - noeartru@cs.toronto.edu

Dataset is very huge (~100GB), and can be shared on request, or if you have access to CS lab clusters you can find it at: "/w/247/abdulbasit/mental_health_dataset_split"
Trained models can also be provided, they couldn't be uploaded to markus.

Description of files: We have kept the training files for the streams seperate so that the reviewers find it easier to run the experiments.

1. analysisScript.py -> this is used to compress the dataset to 500 tweets per user based on lexicons
2. train_mental_bert_classifier.py -> used to train a distilled symptom score classifier
(2.5, optional) test_mental_bert_classifier.py -> verify the distilled model is performing up to par
3. preprocess.py -> uses the distilled mental bert classifier to preprocess user tweets into key information (tweet_id + tweets + symptom_scores + isTopK), and saves it in a parquet file 
4. combined.py -> this is the training and testing file for running the combined model of risky posts and symptom streams with projection vector. 
                Usage for training: python combined.py --mode=train
                Usage for testing python combined.py --mode=test --test_ckpt=/w/331/abdulbasit/mhc_classification/ckpt_combined
5. split_dataset.py -> splits dataset into train, test, and val sets for each disease and control class.
6. risky_posts_stream_experiments.py -> Used to train the risky posts stream with different modifications to weight decay, learning rate decay, and optimizers.
7. GRU_risky_posts_stream.py -> Used to train GRU iteration of risky posts stream with different learning rates.
8. symptom_stream_experiments.py -> Used to train the risky posts stream with different modifications to learning rate decay.
9. GRU_symptom_stream.py -> Used to train GRU iteration of symptom stream with different learning rates.
10. download_images.py -> downloads user tweet images out of urls to serve as training data for the image classifier stream 
11. image_stream.py + image_stream_test.py -> classifies user mental health condition using those images
12. llm_classifier.py + llm_stats.py -> Baseline LLM testing of the project's objective using Llama3.3 

Repository inspired by https://github.com/chesiy/EMNLP23-PsyEx
