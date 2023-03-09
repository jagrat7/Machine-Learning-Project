import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import helper.label_encoder as lenc
df = pd.read_csv("./data/emp_attrition.csv")


def head():
    print(df.head())

#x variable for model , i took variables i thought were important for attrition 
features=df[['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
       'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome',  'NumCompaniesWorked', 'Over18',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]


y = df[['Attrition']]#y variable
Y = df["Attrition"]
def heatmap(df):
    sns.heatmap(df.corr(),cmap="icefire" )
    plt.show()
def histoo(x):
    plt.hist(x,bins=10)
    plt.show()
def scatter(x,y):
        plt.scatter(x,y)
        plt.show()
#check for missing values
def check_missing(df):
    for column in df.isnull().columns.values.tolist():
        print(column)
        print (df.isnull()[column].value_counts())
        print("")  

#graphical analysis 
# histoo(y.values)
# heatmap(df)

#label encoding 
def label_encoder_fuction(df,col_name, data_series_oftarget,*dataframe):
        le =LabelEncoder()

        if (dataframe != None):      
                df["%s_encoded"%(col_name)] = le.fit_transform(data_series_oftarget)
                
        else:        
                print("oo")
                dataframe ["%s_encoded"%(col_name)] = le.fit_transform(data_series_oftarget)
       
        if col_name == "Attrition":
                None
        else:        
                features.drop(col_name,axis= 1,inplace = True)

label_encoder_fuction(df,"Attrition",y)

#encoding features column and dropping
list1 = (features.dtypes == object).tolist()
list2 = features.columns.tolist()
i = len(list1)
z=0
for x in list2:
        if (list1[z] == True):
                label_encoder_fuction(list2[z],features[list2[z]],features)
                features["%s_en"%(list2[z])]= df["%s_encoded"%(list2[z])].values.ravel()
                #features['OverTime_en']= df['OverTime_encoded'].values.ravel()
               
                if (z<i-1):
                        z+=1
                else:
                        break
                
        else:
               
                if (z<i-1):
                        z+=1
                else:
                        break
       

print("\n",'features type:',features.dtypes)
                      
y = df[["Attrition_encoded"]]
Y = df["Attrition_encoded"]

#ONE HOT ENCODING
Nominal_features = df[['Department', 'EducationField', 'Gender' ,'Over18','OverTime','JobRole','MaritalStatus' ]]  #question about maritalstatus and job role

Nominal_features_dummies = pd.get_dummies(Nominal_features)

features = pd.concat([features, Nominal_features_dummies], axis=1)
print (features.head())

list3 = Nominal_features.columns.tolist()
z=0
i=5
for x in features.columns.tolist():
        if x=="%s_en"%(list3[z]):
                features.drop("%s_en"%(list3[z]), axis = 1, inplace=True)
                
                if (z<i-1):
                        z+=1
                else:
                        break

#Normalization
sand_features = df[['Age','DistanceFromHome','DailyRate','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears'
,'TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]]

list4 =sand_features.columns.tolist()
z=0
i=len(sand_features)
print(sand_features.std())
for v in list4:
        if v == list4[z]:
                features[list4[z]]= features[list4[z]]/features[list4[z]].std()
                if (z<i-1):
                        z+=1
                else:
                        break


print (features.head(), features.columns)
features=features.astype('int')

# #checking for unbalanced data
histoo(y.values)             
print (Y.value_counts())    
print("\n",'features type:',features.dtypes)

 #spliting data and using SMOTE and cross validation
smt = SMOTE()
kf = KFold(n_splits=10)


for train_index, test_index in kf.split(features):
        
        x_train, x_val = features.iloc[train_index], features.iloc[test_index]
        
        y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]


#x_train, x_val, y_train, y_val = train_test_split(features, Y, test_size = .2,random_state=12)

x_train_res, y_train_res = smt.fit_sample(x_train, y_train)

print("After SMOTE;")
histoo(y_train_res)             

print (y_train_res)    

# #using SVM as our model 
print('using SVM model')
clf = svm.SVC(kernel='rbf')
clf.fit(x_train_res, y_train_res) 
print(clf.predict(x_val))
print ('Validation Results')
print("accuracy score:",clf.score(x_val, y_val))
print( 'recall_score:',recall_score(y_val, clf.predict(x_val)) )
print('precision_score:',precision_score(y_val, clf.predict(x_val)),'\n')

# #using K nearest neighbours
print('using K nearest neighbours')
neigh = KNeighborsClassifier(n_neighbors = 4).fit(x_train_res, y_train_res)
yhat = neigh.predict(x_val)
print(neigh.predict(x_val))
print ('Validation Results')
print("accuracy score:",neigh.score(x_val, y_val))
print( 'recall_score:',recall_score(y_val, neigh.predict(x_val)) )
print('precision_score:',precision_score(y_val, neigh.predict(x_val)))


