import pandas as pd
import numpy as np

path = '/w/247/baileyng/tweets_preprocessed.parquet'

col_names = ['Do things easily get painful consequences', 'Worthlessness and guilty',
       'Diminished emotional expression', 'Drastical shift in mood and energy',
       'Avoidance of stimuli', 'Indecisiveness',
       'Decreased energy tiredness fatigue', 'Impulsivity',
       'Loss of interest or motivation', 'Fears of being negatively evaluated',
       'Intrusion symptoms', 'Anger Irritability', 'Flight of ideas',
       'Obsession', 'Inattention', 'Compulsions', 'Poor memory',
       'Catatonic behavior', 'Somatic symptoms others', 'Pessimism',
       'Anxious Mood', 'Fear about social situations', 'Respiratory symptoms',
       'More talkative', 'Panic fear', 'Weight and appetite change',
       'Suicidal ideas', 'Depressed Mood', 'Gastrointestinal symptoms',
       'Hyperactivity agitation', 'Somatic symptoms sensory',
       'Autonomic symptoms', 'Genitourinary symptoms', 'Sleep disturbance',
       'Compensatory behaviors to prevent weight gain',
       'Cardiovascular symptoms', 'Somatic muscle', 'Fear of gaining weight']


def read_scores(path, col_names):
    '''
    returns Nx38 matrix
    '''
    df = pd.read_parquet(path)
    matrix = df[col_names].to_numpy()
    return matrix

def read_scores_mean(path, col_names):
    '''
    returns 1x38 matrix. Nx38 is reduced to 1x38 by taking the mean of each column.
    '''
    df = pd.read_parquet(path)
    column_means = df[col_names].mean(axis=0).to_numpy()
    matrix = column_means.reshape(1, -1)
    return matrix

matrix = read_scores(path, col_names)
print("Nx38 Matrix Shape:", matrix.shape)

matrix = read_scores_mean(path, col_names)
print("1x38 Matrix Shape:", matrix.shape)
print(matrix)

df = pd.read_parquet(path)
print(df.shape)
print(df.loc[0, "text"])
print(type(df.loc[0, "text"]))
