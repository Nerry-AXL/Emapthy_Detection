#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import warnings
import re
#import missingno as msno
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold,train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')


# In[2]:


def extract_participant_number(string):
    '''Function to extract participant number from the filename'''
    #Regular expression pattern to match the format 'participant_x'
    pattern = r'participant_(\d+)'
    # Matching the pattern with the filename
    match = re.search(pattern, string)
    # Extracting the participant number and convert it to an integer when match is found
    if match:
        return int(match.group(1))
    # If no match was found, return None
    else:
        return None


# In[3]:


def get_first_nonnull_row(df, col_name):
    # Get a Series of boolean values indicating whether each row has a non-null value in the specified column
    nonnull_mask = df[col_name].notnull()
    # Find the index of the first True value in the Series 
    first_nonnull_idx = nonnull_mask.idxmax()
    return first_nonnull_idx


# In[4]:


def dataselection(df):
    col = 'Pupil diameter left'
    #Finding the 1st row with non null value on pupil diameter left column
    start_index_left = get_first_nonnull_row(df,col)
    col = 'Pupil diameter right'
    #Finding the 1st row with non null value on pupil diameter right column
    start_index_right = get_first_nonnull_row(df,col)
    #Comparing both to find the 1st measured value of pupil diameter
    if start_index_left < start_index_right :
        selected_df = df.iloc[start_index_left::3] # Incrementing 3 rows each time as pupil diameter is measered on 40HQs frequency
    else :
        selected_df = df.iloc[start_index_right::3]
    return selected_df


# In[5]:


def unique_val_cols(df):
    '''Removing Coloumns having are having same values on every rows'''
    unique_values = []
    for column in df.columns :
        if column != 'Participant name' and column != 'Recording timestamp':
            if df[column].nunique() <= 1: # If columns have unique values more than 1, it means that columns have variance
                unique_values.append(column)
    df = df.drop(columns = unique_values) # Removing columns with zero variance
    return df


# In[6]:


def removing_unwanted_coloumns(df, unwanted_cols):
    '''Function to remove unwanted column from the dataset'''
    df = df.drop(columns = unwanted_cols, axis =1)
    return df


# In[7]:


def numeric_conversion(df):
    '''Function to replace comma in dataset III to dot and type casting object datapoints into numerica datapoints'''
    df = df.replace(',','.',regex = True)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df


# In[8]:


def removing_duplicates(df):
    '''Function to remove duplicates from the dataset considering the Eyetracker Time Stamp Coloumn'''
    df = df.drop_duplicates(subset = 'Recording timestamp')
    return df


# In[9]:


def empathy_score(df,df_question):
    '''
       Fuction that takes empathy score which is our target variable from the questionare data set and
       it to the Dataset III.
    '''
    empathy_scores= {} # Empathy score dictonary
    for index, row in df_question.iterrows():
        empathy_scores[int(row['Participant nr'])] = row['Total Score extended'] # Adding empathy to the dictornary as value and participant number as the key
    df['Empathy Score'] = 0
    df['Empathy Score'] = df['Participant name'].apply(lambda x: empathy_scores.get(x, 0)) # Adding Empathy Score to the dataframe based on the participant number
    return df, empathy_scores


# In[10]:


def imputation(df):
    imputer = KNNImputer(n_neighbors=5)
    imputed_df = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns)
    return imputed_df


# In[11]:


def moddata(df):
    
    #Selected coloumns
    col = ['Gaze point X', 'Gaze point Y',
       'Pupil diameter left', 'Pupil diameter right',
       'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)',
       'Eye position left Z (DACSmm)', 'Eye position right X (DACSmm)',
       'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
       'Gaze event duration','Empathy Score']
    try:
        # Creating a numpy array by taking mode values of selected columns
        mode_values = df[col].mode().iloc[0]
    except KeyError as e:
        missing_col = str(e).strip('\'[]\'')
        print(f"Column {missing_col} not found in the dataframe")
        return None
    #Converting numpy array into pandas dataframe
    mode_df = pd.DataFrame([mode_values], columns=col)
    mode_df['Participant name'] = df['Participant name'].iloc[0]
    mode_df['Avg Gaze event duration'] = df['Gaze event duration'].mean()
    mode_df['Total Gaze event duration'] = df['Gaze event duration'].sum()
    return mode_df
    


# In[12]:


def drop_correlation(df):
    cor_matrix = df.corr()
    plt.subplots(figsize = (42,42))
    plt.title('Pearson Corealation Matrix')
    sns.heatmap(cor_matrix,vmax = 0.13,annot=True)
    cor_col = set()
    for i in range(len(cor_matrix.columns)):
        for j in range(i):
            if abs(cor_matrix.iloc[i,j]) > .7 :
                col_n = cor_matrix.columns[i]
                cor_col.add(col_n)
    print('Coloums with Corelation are - ', cor_col)
    df  = df.drop(columns = cor_col, axis =1)
    return df


# In[13]:


def min_max_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    X_scaled = pd.DataFrame(scaler.transform(df),columns=df.columns)
    return X_scaled


# In[14]:


def feature_seperation(df):
    y_df = df['Empathy Score']
    x_df = df.drop(['Empathy Score'], axis= 1)
    return x_df,y_df


# In[15]:


def cross_validation(x_df,y_df,var):
    groups = x_df['Participant name'].tolist()
    n_splits = 10
    gkf = GroupKFold(n_splits = n_splits)
    model = RandomForestRegressor(n_estimators = 10, random_state = 13)
    scores = []
    for train_index, test_index in gkf.split(x_df,y_df, groups = groups):
        x = x_df.drop(['Participant name'],axis=1)
        x_train , x_test = x.iloc[train_index], x.iloc[test_index]
        y_train , y_test = y_df.iloc[train_index], y_df.iloc[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        r2 = 1 - (mse/var)
        print("R2 score -Cross: {:.3f}".format(r2))
        print("MSE- Cross: {:.3f}".format(mse))
        print("Var: {:.3f}".format(var))
        scores.append(r2)

    fig,ax = plt.subplots()
    ax.plot(range(1,n_splits+1),scores, marker ='*')
    ax.set_xlabel('Sample')
    ax.set_ylabel('R-Squared')
    ax.set_title('Cross-Validation Scores')
    plt.show()


# In[16]:


def train_test_split(x_df, y_df, test_size, random_state):
    groups = x_df['Participant name'].tolist()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(x_df, y_df, groups=groups))
    x_train = x_df.iloc[train_idx].drop('Participant name', axis=1)
    x_test = x_df.iloc[test_idx].drop('Participant name', axis=1)
    y_train = y_df.iloc[train_idx]
    y_test = y_df.iloc[test_idx]
    return x_train, x_test, y_train, y_test
    
    


# In[17]:


def ml_model(x_train, x_test, y_train, y_test,var):
    model = RandomForestRegressor(n_estimators=10, random_state=13)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = 1 - (mse/var)
    print("R2 score: {:.3f}".format(r2))
    print("MSE: {:.3f}".format(mse))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(x_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, x_train.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances
    plt.figure(figsize=(10,5))
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices],
        color="r", align="center")
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation='vertical')
    plt.xlim([-1, x_train.shape[1]])
    plt.show()


# In[ ]:





# # Dataloading

# In[18]:


path = r"C:\Users\mg22322\EyeT\EyeT_group_dataset_III_image_name_letter_card_participant_**_trial_*.csv"
df_question =pd.read_csv(r"C:\Users\mg22322\Questionnaire_datasetIB.csv", encoding = 'ISO-8859-1')
filename = glob.glob(path)
df = pd.DataFrame()
for file in filename:
    df_pre = pd.read_csv(file)
    participant_number = extract_participant_number(file)
    df_pre['Participant name'] = participant_number
    selected_df = dataselection(df_pre)
    selected_df = numeric_conversion(selected_df)
    selected_df = removing_duplicates(selected_df)
    unwanted_cols = ['Eye movement type','Computer timestamp','Eye movement type index','Unnamed: 0','Sensor','Event','Event value','Validity left','Validity right','Presented Stimulus name','Presented Media width','Presented Media name','Presented Media height','Presented Media position X (DACSpx)','Presented Media position Y (DACSpx)','Original Media width','Original Media height','Mouse position X','Mouse position Y']
    selected_df = removing_unwanted_coloumns(selected_df, unwanted_cols)
    selected_df = unique_val_cols(selected_df)
    selected_df,empathy_scores = empathy_score(selected_df,df_question)
    selected_df = imputation(selected_df)
    selected_df = moddata(selected_df)
    df = pd.concat([df,selected_df])

df = min_max_scaler(df)
x_df, y_df = feature_seperation(df)
var = np.var(y_df)
x_df = drop_correlation(x_df)
cross_validation(x_df,y_df,var)
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=85)
model = ml_model(x_train, x_test, y_train, y_test,var)


# In[19]:


path = r"C:\Users\mg22322\EyeT\EyeT_group_dataset_II_image_name_grey_blue_participant_**_trial_*.csv"
df_question =pd.read_csv(r"C:\Users\mg22322\Questionnaire_datasetIB.csv", encoding = 'ISO-8859-1')
filename = glob.glob(path)
df = pd.DataFrame()
for file in filename:
    df_pre = pd.read_csv(file)
    participant_number = extract_participant_number(file)
    df_pre['Participant name'] = participant_number
    selected_df = dataselection(df_pre)
    selected_df = numeric_conversion(selected_df)
    selected_df = removing_duplicates(selected_df)
    unwanted_cols = ['Eye movement type','Computer timestamp','Eye movement type index','Unnamed: 0','Sensor','Event','Event value','Validity left','Validity right','Presented Stimulus name','Presented Media width','Presented Media name','Presented Media height','Presented Media position X (DACSpx)','Presented Media position Y (DACSpx)','Original Media width','Original Media height']
    selected_df = removing_unwanted_coloumns(selected_df, unwanted_cols)
    selected_df = unique_val_cols(selected_df)
    selected_df,empathy_scores = empathy_score(selected_df,df_question)
    selected_df = imputation(selected_df)
    selected_df = moddata(selected_df)
    df = pd.concat([df,selected_df])
    
df = min_max_scaler(df)
x_df, y_df = feature_seperation(df)
var = variance = np.var(y_df)
x_df = drop_correlation(x_df)
cross_validation(x_df,y_df,var)
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=45)
ml_model(x_train, x_test, y_train, y_test,var)


# In[ ]:





# In[ ]:




