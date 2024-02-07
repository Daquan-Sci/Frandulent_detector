import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

Train_Inpatientdata = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
Train_Outpatientdata = pd.read_csv('Train_Outpatientdata-1542865627584.csv')
Train_Beneficiarydata = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
Train = pd.read_csv('Train-1542865627584.csv')

merged_Train_Outpatientdata_Train_Beneficiarydata = pd.merge(Train_Outpatientdata, Train_Beneficiarydata, on ='BeneID', how = 'left')
merged_Train_Outpatientdata_Train_Beneficiarydata_Train_Inpatientdata = pd.merge(
    merged_Train_Outpatientdata_Train_Beneficiarydata,
    Train_Inpatientdata,
    on='BeneID',
    how='left',
    suffixes=('', '_inpatient')
)
df = pd.merge(merged_Train_Outpatientdata_Train_Beneficiarydata_Train_Inpatientdata, Train, on ='Provider', how = 'left')


df['PotentialFraud'] = df['PotentialFraud'].apply(lambda x: 1 if x == "Yes" else 0) # 1: is fraud, 0: no fraud

# Assuming df is your merged DataFrame
df = pd.merge(merged_Train_Outpatientdata_Train_Beneficiarydata_Train_Inpatientdata, Train, on='Provider', how='left')

# Convert 'PotentialFraud' column to binary (1 for 'Yes', 0 for 'No')
df['PotentialFraud'] = df['PotentialFraud'].apply(lambda x: 1 if x == "Yes" else 0)

# Drop rows with missing values in 'AttendingPhysician' column
df.dropna(subset=['AttendingPhysician'], inplace=True)

# Now 'df' contains only rows where 'AttendingPhysician' is not NA

# Assuming you've defined the AttendingPhysician_digit function
def AttendingPhysician_digit(x):
    digits = ''
    for char in str(x):
        if char.isdigit():
            digits += char
    return int(digits) 

# Apply AttendingPhysician_digit to 'AttendingPhysician' column
df['AttendingPhysician'] = df['AttendingPhysician'].map(AttendingPhysician_digit)

# Select features and target
features = df[['InscClaimAmtReimbursed', 'County', 'State', 'OPAnnualReimbursementAmt', 'AttendingPhysician']]
target = df['PotentialFraud']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# train the Random Forest model with optimal hyperparmaters
rf_pipe = Pipeline(
            steps=[
            ('rf', RandomForestClassifier(n_estimators = 200,
                                          max_depth = 100,
                                          min_samples_leaf = 5,
                                          max_features = 5,
                                          class_weight = 'balanced'
                                            ))]
                           )

rf_model = rf_pipe.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
