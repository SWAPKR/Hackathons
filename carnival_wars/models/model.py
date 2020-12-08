import pandas as pd 
import numpy as np 
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
import sklearn
from sklearn.svm import SVR
from sklearn import impute
import tsfresh
import seaborn as sns
sns.set(rc={'figure.figsize':(12,8.27)})

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
product_id=test.Product_id
# train.head()


def ncat(x):
    if x<200:
        return 0
    if x>=200 and x<300:
        return 1
    if x>=300 and x<400:
        return 2
    if x>=400:
        return 3


# train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())


train=train[train["Selling_Price"]>0]
train=train[train["Minimum_price"]<=22000]
train=train[train["Maximum_price"]<=33000]


train=train[((train["Selling_Price"]<train["Minimum_price"]) & (train["Discount_avail"]==1.0)) |
            ((train["Selling_Price"]>train["Minimum_price"]) & (train["Discount_avail"]==0.0))]


# train["Selling_Price"]=train["Selling_Price"].fillna(train["Selling_Price"].mean())
test.loc[:, "Selling_Price"] = -1000000000
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
data=data.drop(columns=['Customer_name','Product_id'])
# data=data.drop('instock_date')
# data=data.drop(' Product_id')






a=data["Minimum_price"].mean()
b=data["Maximum_price"].mean()
c=b-a



data['Grade_chg2']=data.Grade.astype(str) + '_' + data['charges_2 (%)'].astype(str)
# data['stall_prod']=data.Product_Category + '_' + data.Stall_no.astype(str)
data['prod_grd']=data.Product_Category + '_' + data.Grade.astype(str)
# data['prod_demand']=data.Product_Category + '_' + data.Demand.astype(str)

data["charges_1"]=data["charges_1"].fillna(data["charges_1"].mean())
data["charges_range"]=data["charges_1"].apply(ncat)

# data["charges_2 (%)"]=data["charges_2 (%)"].fillna(data["charges_2 (%)"].mean())


data["Minimum_price"]=data["Minimum_price"].fillna(data["Maximum_price"]-c)
data["Maximum_price"]=data["Maximum_price"].fillna(data["Minimum_price"]+c)

data["Minimum_price"]=data["Minimum_price"].fillna(data["Minimum_price"].mean())
data["Maximum_price"]=data["Maximum_price"].fillna(data["Maximum_price"].mean())

# "charges_1","charges_2 (%)"","Minimum_price","Maximum_price","Selling_Price"

data['mean_p']=(data['Minimum_price']+data['Maximum_price'])//2


# data.instock_date=pd.to_datetime(data.instock_date)
# data['year']=data.instock_date.dt.year
# data['month']=data.instock_date.dt.month
# data['weekday']=data.instock_date.dt.dayofweek
data.drop('instock_date',axis=1,inplace=True)


num_col=['Maximum_price','Minimum_price','Selling_Price','charges_1','mean_p','Demand']
cat_col=[col for col in data.columns if col not in num_col]

for feat in cat_col:
    lbl_enc = preprocessing.LabelEncoder()
    temp_col = data[feat].fillna("None").astype(str).values
    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
train2= data[data.Selling_Price != -1000000000].reset_index(drop=True)
test2 = data[data.Selling_Price == -1000000000].reset_index(drop=True)


# print(train2.info())
# print(test2.info())

# print(train2.describe())
# print(test2.describe())




x_test=test2.drop('Selling_Price',axis=1)




x_train=train2.drop('Selling_Price',axis=1)
y_train=train2["Selling_Price"].astype(float)



# x_train, x_vld, y_train, y_vld = train_test_split(x_train,y_train test_size=0.2, random_state=42)

# trainx=np.array(x_train)
# valx=np.array(x_test)
# testx=np.array(testx)




model=ensemble.GradientBoostingRegressor(n_estimators=160,max_depth=7)
# model=XGBRegressor(max_depth=15,n_estimators=155)
# model=ensemble.GradientBoostingRegressor(n_estimators=370,max_depth=8,min_samples_split=2,learning_rate=0.035,loss='ls')


# model=ensemble.GradientBoostingRegressor(n_estimators=305,max_depth=5,min_samples_split=2,learning_rate=0.05,loss='ls')
# model=ensemble.GradientBoostingRegressor(n_estimators=290,max_depth=5,min_samples_split=2,learning_rate=0.04,loss='ls')
# model=sklearn.ensemble.AdaBoostRegressor(base_estimator=None, n_estimators=200, learning_rate=0.1, loss='square', random_state=2)
# model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
#  model=sklearn.linear_model.Ridge()
# model=sklearn.linear_model.Lasso()
# model=sklearn.linear_model.ElasticNet()
# model=sklearn.linear_model.LinearRegression()
# model=sklearn.linear_model.LogisticRegression()
# model=SVR(kernel='rbf')
# model=sklearn.linear_model.SGDRegressor(loss='squared_loss', alpha=0.001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.01, shuffle=True, verbose=2, epsilon=0.1, random_state=2, learning_rate='optimal', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)


# fit model on training data (ohe)
print("training your model...")
model.fit(x_train,y_train)





print(model.score(x_train,y_train))
# print(model.score(x_vld,y_vld))




# a=model.predict(trainx)
# # print(a)
# b=model.predict(valx)
# # print(b)
result=model.predict(x_test)


# def root_mean_squared_error(y_true, y_pred):
#         return np.sqrt(np.mean(np.square(y_pred - y_true)))
# print("training loss: ",root_mean_squared_error(np.array(y_train),a))
# # print("val loss: ",root_mean_squared_error(np.array(y_test),b))




submission=pd.Series(data=result,index=product_id,name='Selling_Price')
submission[submission<0]=submission[submission<0]*(-1)
submission.to_csv('../prediction/3.csv')