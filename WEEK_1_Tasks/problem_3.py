import pandas as pd
import numpy as np

names=['stud1', 'stud2', 'stud3', 'stud4', 'stud5', 'stud6', 'stud7', 'stud8', 'stud9', 'stud10']
subject=['sub1', 'sub2', 'sub1', 'sub2', 'sub3', 'sub4', 'sub3', 'sub4', 'sub5', 'sub1']
scores=np.random.randint(50, 101, size=10)

data={
    'Name': names,
    'Subject': subject,
    'Score': scores,
    'Grade': ''
}
df=pd.DataFrame(data)

def grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade']=df['Score'].apply(grade)
df.sort_values('Score', ascending=False)
print(df)
df1=df.groupby('Subject')['Score'].mean()
print(df1)

def pandas_filter_pass(dataframe):
    res=df[df['Grade'].isin(['A', 'B'])]
    return res

print(pandas_filter_pass(df))