from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
class project:

    def __init__(self,config):
        self.config=config
        pass
    def handel_data(self,dx,istrain=False,isval=False,istest=False):
        df=dx.copy()
        df["pickup_datetime"]=pd.to_datetime(df["pickup_datetime"])
        df["pickup_hour"]=df["pickup_datetime"].dt.hour
        df["pickup_minute"]=df["pickup_datetime"].dt.minute
        df["pickup_month"]=df["pickup_datetime"].dt.month
        df["is_weekend"]=df["pickup_datetime"].dt.dayofweek.isin([4,5])
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'  
            elif month in [3, 4, 5]:
                return 'Spring'  
            elif month in [6, 7, 8]:
                return 'Summer'  
            else:
                return 'Fall'    
        df['season'] = df['pickup_month'].apply(get_season)
        
        
        def Haversine_func(long1,lati1,long2,lati2):
            lati1,lati2=np.radians(lati1),np.radians(lati2)
            long=np.radians(long2-long1)
            lati=lati2-lati1
            a=np.sin(lati/2)**2+np.cos(lati1)*np.cos(lati2)*np.sin(long/2)**2
            c=2*np.arcsin(np.sqrt(a))
            d=c*6371.0
            return d
        def Manhattan_func(long1,lati1,long2,lati2):
            r=6371.0
            long1,lati1,long2,lati2=map(np.radians,[long1,lati1,long2,lati2])
            return (np.abs(lati1-lati2)*r+np.abs(long1-long2)*r*np.cos((lati1+lati2)/2))
        df["Haversine_Distance"]=Haversine_func(df["pickup_longitude"],df["pickup_latitude"],df["dropoff_longitude"],df["dropoff_latitude"])
        df["Manhattan_Distance"]=Manhattan_func(df["pickup_longitude"],df["pickup_latitude"],df["dropoff_longitude"],df["dropoff_latitude"])
        df["is_weekend"]=df["is_weekend"].astype("int")
        df['store_and_fwd_flag'] =(df['store_and_fwd_flag'].str.lower().eq('y').astype("int"))
        df=pd.get_dummies(df,columns=["season"],drop_first=False,dtype=int)
        
        def remove_outliers(log_duration):
            q1 = log_duration.quantile(0.25)
            q3 = log_duration.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            clean_mask = ((log_duration>= lower_bound) & (log_duration <= upper_bound))
            
            df_clean = df[clean_mask].copy()
            return df_clean
        df["lati_diff"]=np.abs(df["dropoff_latitude"]-df["pickup_latitude"])
        df["long_diff"]=np.abs(df["dropoff_longitude"]-df["pickup_longitude"])
        df["area_bounding_box"]= df["lati_diff"]* df["long_diff"]
        df["distance_rathio"]=(df["Manhattan_Distance"]/(df["Haversine_Distance"]+0.001))
        df["distance_diff"]=df["Manhattan_Distance"]-df["Haversine_Distance"]
        df["long_diff"]=np.log1p(df["long_diff"])
        df["lati_diff"]=np.log1p(df["lati_diff"])
        df["Haversine_Distance"],df["Manhattan_Distance"]=np.log1p(df["Haversine_Distance"]),np.log1p(df["Manhattan_Distance"])
        df['sin_hour'] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)
        df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        df['Manhattan intersect Haversine'] =df["Manhattan_Distance"]*df["Haversine_Distance"]
        df["pickup_day"]=df["pickup_datetime"].dt.dayofweek
        df=df.drop(columns=["vendor_id","id","pickup_datetime"],errors='ignore')
                
        if istrain :
            df["trip_duration"]= np.log1p(df["trip_duration"])
            df=remove_outliers(df["trip_duration"])
            t=df["trip_duration"]
            df=df.drop(columns=("trip_duration"))
        elif isval or istest:
            df["trip_duration"]= np.log1p(df["trip_duration"])
            t=df["trip_duration"]
            df=df.drop(columns=("trip_duration"))
        else :t=None
        return df,t
    
    def evaluate_model(self,data):
        data=pd.read_csv(data)
        model=Ridge(alpha=1)
        df=pd.read_csv(self.config["data"])
        train_data,train_gt=self.handel_data(df,istrain=True)
        scaler=StandardScaler()
        train_data=scaler.fit_transform(train_data)
        self.data=data.copy()
        data_train,data_gt=self.handel_data(data,isval=True)
        data_train=scaler.transform(data_train)
        model.fit(train_data,train_gt)
        prediction= model.predict(data_train)
        print(f"Rmse for the this data  {root_mean_squared_error(data_gt,prediction)}")
        print(f"r2 score for this data   {r2_score(data_gt,prediction)}")
       
        