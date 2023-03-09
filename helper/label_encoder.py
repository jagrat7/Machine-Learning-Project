
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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


