import pandas as pd
from sklearn.metrics import classification_report

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "mdd",
    "neg",
    "ocd",
    "ppd",
    "ptsd"
]

verbose2ShortDisease = {
    "Attention Deficit Hyperactivity Disorder": "adhd",
    "Anxiety": "anxiety",
    "Bipolar Disorder": "bipolar",
    "Depression": "depression",
    "Major Depressive Disorder": "mdd",
    "Negative (Control group)": "neg",
    "Obsessive Compulsive Disorder": "ocd",
    "Postpartum Depression": "ppd",
    "Post-Traumatic Stress Disorder": "ptsd"
}


if __name__ == "__main__":
    df1 = pd.read_csv('arliai_predictions200.csv')
    df2 = pd.read_csv('arliai_predictions400.csv')

    df = pd.concat([df1, df2], ignore_index=True)

    verbose2ShortDisease = {k.lower().replace(" ", ""): v for k, v in verbose2ShortDisease.items()}

    
    df['pred'] = df['pred'].str.lower().str.replace(" ", "").map(verbose2ShortDisease)
    
    df['true'] = df['true']

    report = classification_report(df['true'], df['pred'], target_names=id2disease)
    with open('classification_report.txt', 'w') as f:
        f.write(report)