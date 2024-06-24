from django.shortcuts import render,HttpResponse ,redirect
from django.http import HttpResponseNotFound
import plotly.express as px
from datetime import timedelta,datetime
from .forms import DateForm

import pandas as pd
import numpy as np 
import hydroeval as hy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import random
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error

from hyperopt import hp, tpe, Trials, fmin
from tensorflow.keras.optimizers import Adam

def temp(request):

    return render(request,'model_op.html',{'r2':1})


def classification_view(request):
    return render(request,'classification.html')

from .models import RegressionModel
def regression_view(request):

    if request.method == "POST":
        model_name = request.POST.get('reg_task', '')

        data_name=request.POST.get('data_format','D')

        stop_criteria=request.POST.get('stop','rmse')
        seasonality=request.POST.get('season',True)
        
        csv_file= request.FILES.get('csv_name', '')

        print(model_name,data_name,stop_criteria,seasonality,csv_file,sep='=>')

        df=pd.read_csv(csv_file,parse_dates=True)
        test_size=int(df.shape[0]*0.2)


        tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)
        i = 1
        score = []
        for fold_number, (tr_index, val_index) in enumerate(tscv.split(df)):
            if i <= 1:
                print(fold_number+1)
                print(tr_index, val_index)
            i += 1
        
        ts_column = 'Date'
        sep = ','
        target = 'discharge'

        traindata = df.iloc[tr_index]
        testdata = df.iloc[val_index]

        print('its seanoality type ',type(seasonality))
        s=False
        if seasonality=='on':
            s=True
        model = auto_timeseries(forecast_period=test_size,score_type=stop_criteria,
                time_interval=data_name,# for hour H, month M
                non_seasonal_pdq=None, seasonality=s, seasonal_period=12,
                model_type=[model_name] # model type Prophet,ML,VAR, auto_SARIMAX, Best
                ,dask_xgboost_flag=False,
                verbose=2)
        
        print(traindata.shape,testdata.shape)
        model.fit(traindata, ts_column,target)

       
        predictions  = model.predict(testdata=testdata, model=model_name)
        print(predictions.shape)
        return HttpResponse('predictions')
        # tes_pred = testdata[[target]]
        
        # tes_pred['yhat'] = predictions['yhat'].values
        # print('prediction yhat')
        # print(tes_pred['yhat'].values)
        # ((pd.concat([tes_pred[target], tes_pred['yhat']],axis=1).dropna(axis=1)).plot(figsize=(15,7))).savefig('BtpProject/static/plots/pred.png')
        
        # automl_metrics=print_ts_model_stats(tes_pred[target], tes_pred['yhat'])    


        

        # observed_values = tes_pred[target].values
        # simulated_values = tes_pred['yhat'].values

        # # Calculate Nash-Sutcliffe Efficiency (NSE)
        # nse = hydroeval.nse(observed_values, simulated_values)

        # # Calculate Kling-Gupta Efficiency (KGE)
        # kge = hydroeval.kge(observed_values, simulated_values)



        # return render(request,'regression_op.html',{'model':model,'metric':automl_metrics,'nse':nse,'kge':kge,'model_name':model_name})
            
    return render(request,'regression.html')


#plotly check 

def plotly_view(request):

    df=pd.read_csv(r'C:\Users\Shubham\Documents\Academics\sem8\BTPI\btp_project\data.csv')
    fig=px.line(df,x='Date',y='discharge')



    return fig

def evaluate_model_df(model, X_train, y_train, X_test, y_test):
  
    # Make predictions on training and test sets
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

   
    # Calculate metrics
    kge_train,r_train,alpha_train,beta_train=hy.kge(y_train, y_pred_train).flatten()
    metrics_train = {
        "R2": r2_score(y_train, y_pred_train),
        "MAPE": mean_absolute_percentage_error(y_train, y_pred_train),
        "MSE": mean_squared_error(y_train, y_pred_train),
        "NSE": hy.nse(y_train, y_pred_train),
        "KGE": kge_train
    }

    kge_test,r_test,alpha_test,beta_test=hy.kge(y_test, y_pred_test).flatten()
    metrics_test = {
        "R2": r2_score(y_test, y_pred_test),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "NSE": hy.nse(y_test, y_pred_test),
        "KGE": kge_test
    }
    # print('shape of y_pred_trian and test is ',y_pred_test.shape,y_pred_train.shape,sep=' => ')
    # print(metrics_train,'this is training metrics ')
    metrics_train_rounded = {key: round(value, 3) for key, value in metrics_train.items()}
    metrics_test_rounded = {key: round(value, 3) for key, value in metrics_test.items()}


    # col1={"R2": "R2",
    #     "MAPE": "MAPE",
    #     "MSE": "MSE",
    #     "NSE":"NSE",
    #     "KGE":"KGE"}
    col1 = {
    "R2": "Coefficient of Determination (R²)",
    "MAPE": "Mean Absolute Percentage Error (MAPE)",
    "MSE": "Mean Squared Error (MSE)",
    "NSE": "Nash-Sutcliffe Efficiency (NSE)",
    "KGE": "Kling-Gupta Efficiency (KGE)"
}

    df = pd.DataFrame([col1,metrics_train_rounded, metrics_test_rounded]).T
    # Create the DataFrame with transposed metrics
    # df = pd.DataFrame([col1,metrics_train, metrics_test]).T
    df.index.name = "Metric"  # Set index name
    df.columns = ["metrics ","Train", "Test"]  # Set column names
    # df['performance']=np.array(["Coefficient of Determination (R²)", "Mean Absolute Percentage Error (MAPE)", "Mean Squared Error (MSE)", "Nash-Sutcliffe Efficiency (NSE)", "Kling-Gupta Efficiency (KGE)"])
    # df.drop(['metrics'],axis=1,inplace=True)
    return df

    
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def lstm_bayesian(raw_seq,train_len):

    lstm_space = {
        'n_steps': hp.choice('n_steps', [3, 5, 7]),
        'n_features': 1,
        'n_units': hp.choice('n_units', [50, 100]),
        'activation': hp.choice('activation', ['relu', 'tanh']),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.01)
    }

    def lstm_hyperopt(params):
        n_steps = params['n_steps']
        n_features = params['n_features']
        n_units = params['n_units']
        activation = params['activation']
        learning_rate = params['learning_rate']

        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        X_train = X[:train_len, :]
        X_test = X[train_len:, :]
        y_train = y[:train_len]
        y_test = y[train_len:]

        # define model
        model = Sequential()
        model.add(LSTM(n_units, activation=activation, input_shape=(n_steps, n_features)))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)


        # model.fit(X_train, y_train, epochs=10, verbose=1)
        model.fit(X_train, y_train, validation_split=0.3,epochs=5, verbose=0,callbacks=[early_stopping])


        # model.fit(X_train, y_train, epochs=1, verbose=0)

        yhat = model.predict(X_test, verbose=0)

        return -r2_score(y_test, yhat.flatten())
    

    # Run Bayesian Optimization
    trials = Trials()
    best = fmin(fn=lstm_hyperopt, space=lstm_space, algo=tpe.suggest, max_evals=1, trials=trials,rstate=np.random.default_rng(42))

    best_params = {
        'n_steps': [3, 5, 7][best['n_steps']],
        'n_features': 1,
        'n_units': [50, 100][best['n_units']],
        'activation': ['relu', 'tanh'][best['activation']],
        'learning_rate': best['learning_rate']
    }
    print('best params in lstm are',best_params)
    return best_params

def ann_bayesian(raw_seq,train_len):
    def ann_objective(params):
                        
        n_units = params['n_units']
        learning_rate = params['learning_rate']
        n_steps= params['n_steps']
        n_features = params['n_features']


        X, y = split_sequence(raw_seq, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        X_train = X[:train_len, :]
        X_test = X[train_len:, :]
        y_train = y[:train_len]
        y_test = y[train_len:]

        

        # Define and compile the ANN model
        ann_model = Sequential()
        ann_model.add(Dense(units=n_units, activation='relu', input_shape=(n_steps,)))
        ann_model.add(Dense(units=n_units, activation='relu'))
        ann_model.add(Dense(units=1))

        optimizer = Adam(learning_rate=learning_rate)
        ann_model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)

        # Fit the model
        ann_model.fit(X_train.reshape((X_train.shape[0], n_steps)), y_train,validation_split=0.3, verbose=0, epochs=500,callbacks=[early_stopping])
        
        # Predict on the test set and calculate R^2 score
        yhat = ann_model.predict(X_test.reshape((X_test.shape[0], n_steps)), verbose=0)
        r2 = r2_score(y_test, yhat.flatten())

        return -r2  # Hyperopt minimizes the objective function, so negate R^2

    # Define the search space for Hyperopt
    ann_space = {
        'n_units': hp.choice('n_units', [8,16,32]),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
        # 'learning_rate': hp.choice('learning_rate', [0.001]),
        'n_steps':  hp.choice('n_steps', [3, 5, 7]),
        'n_features': 1,

    }

    # Run Bayesian Optimization
    trials = Trials()
    best = fmin(fn=ann_objective, space=ann_space, algo=tpe.suggest, max_evals=1, trials=trials,rstate=np.random.default_rng(42))

    # print("Best hyperparameters:", best)

    # Use the best hyperparameters to build the final ANN model
    best_params = {
        'n_units': [8,16,32][best['n_units']],
        'learning_rate':best['learning_rate'],
        'n_steps': [3, 5, 7][best['n_steps']],
        'n_features': 1,
        


    }
    return best_params         

def rf_bayesian(raw_seq,train_len):

    def rf_objective(params):

                    
        n_estimators = params['n_estimators']
        max_features = params['max_features']
        n_steps = params['n_steps']

        n_features=params['n_features']

        X, y = split_sequence(raw_seq, n_steps)


        X_train = X[:train_len, :]
        X_test = X[train_len:, :]
        y_train = y[:train_len]
        y_test = y[train_len:]


        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=1)
        rf_model.fit(X_train, y_train)
        score = rf_model.score(X_test, y_test)

        return -score  # Hyperopt minimizes the objective function, so negate score
    #lag 1 to 10 
    rf_space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
        'max_features': hp.choice('max_features', [2, 3, 4]),
        'n_steps':  hp.choice('n_steps', [3, 5, 7]),
        'n_features': 1,

    }
    
    rf_trials = Trials()
    rf_best = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=1, trials=rf_trials,rstate=np.random.default_rng(42))
    rf_best_params = {
        'n_estimators': [50, 100, 150][rf_best['n_estimators']],
        'max_features': [2, 3, 4][rf_best['max_features']],
        'n_steps': [3, 5, 7][rf_best['n_steps']],
        'n_features': 1
    }

    return rf_best_params





def svm_bayesian(raw_seq,train_len):
    def svm_objective(params):
        C = params['C']
        kernel = params['kernel']
        n_steps = params['n_steps']
        n_features=params['n_features']

        X, y = split_sequence(raw_seq, n_steps)
        X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]

        svr = SVR(C=C, kernel=kernel)
        svr.fit(X_train, y_train)
        score = svr.score(X_test, y_test)

        return -score

    svm_space = {
        'C': hp.uniform('C', 0.1, 10),
        'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly']),
        'n_steps':  hp.choice('n_steps', [3, 5, 7]),
        'n_features': 1,
    }

    svm_trials = Trials()
    svm_best = fmin(fn=svm_objective, space=svm_space, algo=tpe.suggest, max_evals=1, trials=svm_trials,rstate=np.random.default_rng(42))

    # print("Best SVM hyperparameters:", svm_best)

    # Use the best hyperparameters to build the final SVM model
    svm_best_params = {
        'C': svm_best['C'],
        'kernel': ['linear', 'rbf', 'poly'][svm_best['kernel']],
        'n_steps': [3, 5, 7][svm_best['n_steps']],
        'n_features': 1,
    }

    return svm_best_params

def all_info_save(df):
    from scipy.stats import skew, kurtosis
    start_date=df['date'].values[0]
    end_date=df['date'].values[-1]
    average = df['discharge'].mean()

    # Calculate standard deviation
    std_dev = df['discharge'].std()

    # Calculate skewness
    skewness = skew(df['discharge'])

    # Calculate kurtosis
    kurt = kurtosis(df['discharge'])

    # Calculate minimum and maximum values
    minimum = df['discharge'].min()
    maximum = df['discharge'].max()

    # Calculate number of zeros
    num_zeros = (df['discharge'] == 0).sum()

    # Calculate percentage of zero rows
    percent_zero_rows = (num_zeros / len(df)) * 100
    # df['date']=df['date'].dt.strftime('%Y-%m-%d')

    metrics_data = {
        'Properties': ['Start Date', 'End Date', 'Average', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Minimum', 'Maximum', 'Number of Zeros', '% of Zero Rows'],
        'Value': [start_date, end_date, round(average, 3), round(std_dev, 3), round(skewness, 3), round(kurt, 3), minimum, maximum, num_zeros, round(percent_zero_rows, 3)]

    }
    
    df_info=pd.DataFrame(metrics_data)
    df_info.to_csv('csv_files/all_info.csv',index=None)
    return df_info
import os
def home_page(request):
    
    if request.method=='POST':
        photo = request.FILES['img_name']
        # file_path = os.path.join('static', 'btp.jpg')
        data=ImageClass(photo=photo)
        data.save()
        return redirect('/')

        # return redirect('/')

    return render(request,'home.html')


def model_selection_view(request):

    if request.method =='POST':
        csv_file = request.FILES.get('csv_name') 
        date_format = request.POST.get('date_format')
        date_sep = request.POST.get('date_sep')
        date_format=date_format.replace('/',date_sep)
        

        if not csv_file:
            return HttpResponse('file not found ; Upload file again !')
        file_name=csv_file.name

        if 'csv' in file_name:
            print('csv')
            df=pd.read_csv(csv_file)
        elif ('xls' in file_name) or ('xlsx' in file_name):
            print('excel file')
            df=pd.read_excel(file_name)
        else:
            return HttpResponseNotFound(f'{file_name} => this  file format not allowed , allowed files formats are .csv , .xls , .xlsx ')

        if len(df.columns)>2:
            return HttpResponse(f'more than two (2) columns are not allowed')
         # Get the uploaded CSV file
        df.columns=['date','discharge']
        df_info=all_info_save(df)
        df.to_csv('csv_files/data.csv',index=None)
        with open('csv_files/date.txt','w') as f:
            f.write(date_format)
        return render(request,'model_select.html',{'df_all':df_info})

    df_info=pd.read_csv('csv_files/all_info.csv')
    # return HttpResponse('model selection page not found !')
    return render(request,'model_select.html',{'df_all':df_info})

def data_view(request):
    
    if request.method == 'POST':
        # Collect form data

        ml_task = request.POST.get('ml_task')  # Get the selected regression task
        train_split = request.POST.get('train_split')  # Get the selected train split
        n_days=int(request.POST.get('ndays'))
        # seasonality = request.POST.get('season')  # Check if seasonality checkbox is checked
        # csv_file = request.FILES.get('csv_name') 
        with open('csv_files/date.txt','r') as f:
            date_format = f.read()
        # date_sep = request.POST.get('date_sep')

        # date_format=date_format.replace('/',date_sep)
        
       
        df=pd.read_csv('csv_files/data.csv')
       
        percentage_train=10

        percentage_test=100-percentage_train

        

        df['date']=pd.to_datetime(df['date'],format=date_format)

        print(df['date'].dtype)
        df.set_index('date',drop=True,inplace=True)
        dates=df.index
        print(dates[-1],type(dates[-1]))
        
        future_ndates=[dates[-1] + timedelta(days=i) for i in range(1, n_days+1)]

        # return HttpResponse(f'{percentage_test},{df.columns[0]} {df["date"].dtype}hi')
        train_len=int(df.shape[0]*int(train_split)/100)

        with open('csv_files/imp_dates.txt','w') as f:
            f.write(",".join(date_i.strftime('%d/%m/%Y') for date_i in [dates[0],dates[train_len],dates[-1]]))

        # start_date=dates[0]
        # mid_date=dates[train_len]
        # end_date=dates[-1]

        if ml_task=='ann' or ml_task=='lstm':
            
            raw_seq=df['discharge'].values
            
            if ml_task=='lstm':

                
                best_params=lstm_bayesian(raw_seq,train_len)
                n_steps = best_params['n_steps']
                n_features = best_params['n_features']
                n_units = best_params['n_units']
                activation = best_params['activation']
                learning_rate = best_params['learning_rate']

                # split into samples
                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]
                
                # define model
                model = Sequential()
                model.add(LSTM(n_units, activation=activation, input_shape=(n_steps, n_features)))
                model.add(Dense(1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)


                # model.fit(X_train, y_train, epochs=10, verbose=1)
                model.fit(X_train, y_train, validation_split=0.3,epochs=500, verbose=0,callbacks=[early_stopping])

            else:

                best_params=ann_bayesian(raw_seq,train_len)
                n_features = best_params['n_features']
                n_steps = best_params['n_steps']
            
                n_units = best_params['n_units']
            
                learning_rate = best_params['learning_rate']
            

                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))
                X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]
                
                model = Sequential()
                model.add(Dense(units=n_units, activation='relu', input_shape=(n_steps,)))
                model.add(Dense(units=n_units, activation='relu'))
                model.add(Dense(units=1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)

                # Fit the final model
                model.fit(X_train.reshape((X_train.shape[0], best_params['n_steps'])), y_train, validation_split=0.3,verbose=0, epochs=500,callbacks=[early_stopping])

        else:

            raw_seq=df['discharge'].values

            
            if ml_task=='svm':
                
                svm_best_params=svm_bayesian(raw_seq,train_len)
                n_features=svm_best_params['n_features']
                C=svm_best_params['C']
                kernel=svm_best_params['kernel']
                n_steps = svm_best_params['n_steps']
                
                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))
                X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]
                model = SVR(C=C, kernel=kernel)
                X_train=np.squeeze(X_train,axis=-1)
                X_test=np.squeeze(X_test,axis=-1)
                print(f'X_train shape {X_train.shape} {X_test.shape},{y_train.shape},{y_test.shape}')
                # return HttpResponse(X_train.shape)
                model.fit(X_train, y_train)

            else:

                rf_best_params=rf_bayesian(raw_seq,train_len)

                n_estimators=rf_best_params['n_estimators']
                max_features=rf_best_params['max_features']
                n_steps = rf_best_params['n_steps']
                
                X, y = split_sequence(raw_seq, n_steps)
                X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]
                model = RandomForestRegressor(n_estimators=n_estimators,
                                                max_features=max_features,
                                                random_state=1)
                model.fit(X_train, y_train)

            
            model.fit(X_train,y_train)

        model_dict={'lstm':" LSTM ",'ann':"Artificial Neural Network (ANN)",'rf':'Random Forest','svm':"Support Vector Machine"}
        pred_train=model.predict(X_train)
        pred=model.predict(X_test)

        dates=dates[n_steps:]

        date_train=dates[:train_len]
        date_test=dates[train_len:]
        
        print(f'shapes :train {pred_train.shape} test {pred.shape}  {date_train.shape} {date_test.shape}')

        df_metric=evaluate_model_df(model, X_train, y_train, X_test, y_test)
    
        # plt.plot(date_train[::300],y_train[::300],label='actual values')
        # plt.plot(date_train[::300],pred_train[::300],c='r',label='prediction')
        # plt.title(model_dict[ml_task]+" Train")
        # plt.xlabel('per 300 days dates')
        # plt.ylabel('discharge (m3/h)')
        # plt.xticks(rotation=40)
        
        # plt.tight_layout()

        # plt.legend()
        # plt.savefig('BtpProject/static/plots/ml_train.png')
        # plt.close()
        print(pred_train.shape,f'pred train {date_train.shape} date_train  {y_train.shape} y_train,for lstm')
        
        pd.DataFrame({'date':date_train,'y_train':y_train,'pred_train':pred_train.flatten()}).to_csv(f'csv_files/train.csv',index=None)
        pd.DataFrame({'date':date_test,'y_test':y_test,'pred_test':pred.flatten()}).to_csv(f'csv_files/test.csv',index=None)
        # plt.figure(figsize=(5,7))
        # fig.autofmt_xdate()
        # plt.plot(date_test[::200],y_test[::200],label='actual values')
        # plt.plot(date_test[::200],pred[::200],c='r',label='prediction')
    
        # plt.title(model_dict[ml_task]+" Test")
        # plt.xlabel('per 200 days dates')
        # plt.ylabel('discharge (m3/h)')
        # plt.xticks(rotation=40)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('BtpProject/static/plots/ml_test.png')
        # plt.close()
        future_pred = []
        last_n = X_test[-1].copy()

        for i in range(n_days):
            if ml_task == 'ann' or ml_task == 'lstm':
                y_pred = model.predict(last_n.reshape(1, n_steps, 1)).flatten()[0]
            else:
                y_pred = model.predict(last_n.reshape(1, n_steps)).flatten()[0]
            
            future_pred.append(y_pred)
            last_n = np.append(last_n[1:], y_pred)

        
        print(f'lenth of n_dates {len(future_ndates)} lenth of future pred {len(future_pred)}')

        df_pred=pd.DataFrame({'date':future_ndates,'pred':future_pred})
        df_pred.to_csv(f'csv_files/pred.csv',index=None)
        with open('csv_files/imp_dates.txt','r') as f:
            date_arr=f.read().split(',')
        df_metric.columns=['performance metric',f'train data from {date_arr[0]} to {date_arr[1]}',f'test data from {date_arr[1]} to {date_arr[2]}']
        df_metric.to_csv(f'csv_files/metric.csv',index=None)
        # fig_train = px.scatter(x=date_train, y=y_train)
        # plt.plot(df_pred['Date'],df_pred['prediction'],label='predicted values')
        
    
        # plt.title(model_dict[ml_task]+" prediction")
        # plt.xlabel('1 month')
        # plt.ylabel('discharge (m3/h)')
        # plt.xticks(rotation=40)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('BtpProject/static/plots/ml_pred.png')
        # plt.close()
        df1=pd.read_csv('csv_files/metric.csv')

        return render(request,'model_op.html',{'df':df1,'model':model_dict[ml_task]})
            
        
    return render(request,'model_select.html')

def temp_op(request):
    df=pd.read_csv('csv_files/metric.csv')
    return render(request,'model_op.html',{'df':df})
import plotly.express as px

# Create your views here.
def plotly_train(request):

    df=pd.read_csv('csv_files/train.csv')
    start = request.GET.get('start')
    end = request.GET.get('end')

    df['date']=pd.to_datetime(df['date'])

    if start and end:
        # Convert start and end dates to datetime objects
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Filter DataFrame based on date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        # If start or end date is not provided, use the entire DataFrame
        filtered_df = df

    fig = px.line(
        filtered_df,  # Use the filtered DataFrame for plotting
        x='date',     # Assuming 'date' is the column name for dates
        y='y_train',  # Assuming 'average' is the column name for CO2 PPM
        title="Train data ",
        labels={'date': 'Date', 'discharge': 'discharge label'}
    )
    fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_train'], mode='lines', name='Predicted discharge')

    fig.update_layout(
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
        }
    )

    chart = fig.to_html()
    context = {'chart': chart, 'form': DateForm()}  # Assuming DateForm is passed to the context
    return render(request, 'plotly_train.html', context)
def plotly_test(request):

    df=pd.read_csv('csv_files/test.csv')
    start = request.GET.get('start')
    end = request.GET.get('end')

    df['date']=pd.to_datetime(df['date'])

    if start and end:
        # Convert start and end dates to datetime objects
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Filter DataFrame based on date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        # If start or end date is not provided, use the entire DataFrame
        filtered_df = df

    fig = px.line(
        filtered_df,  # Use the filtered DataFrame for plotting
        x='date',     # Assuming 'date' is the column name for dates
        y='y_test',  # Assuming 'average' is the column name for CO2 PPM
        title="test data ",
        labels={'date': 'Date', 'discharge': 'discharge label'}
    )
    fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_test'], mode='lines', name='Predicted discharge')

    fig.update_layout(
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
        }
    )

    chart = fig.to_html()
    context = {'chart': chart, 'form': DateForm()}  # Assuming DateForm is passed to the context
    return render(request, 'plotly_train.html', context)
def plotly_pred(request):

    df=pd.read_csv('csv_files/pred.csv')
    start = request.GET.get('start')
    end = request.GET.get('end')

    df['date']=pd.to_datetime(df['date'])

    if start and end:
        # Convert start and end dates to datetime objects
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Filter DataFrame based on date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        # If start or end date is not provided, use the entire DataFrame
        filtered_df = df

    fig = px.line(
        filtered_df,  # Use the filtered DataFrame for plotting
        x='date',     # Assuming 'date' is the column name for dates
        y='pred',  # Assuming 'average' is the column name for CO2 PPM
        title="prediction data ",
        labels={'date': 'Date', 'discharge': 'discharge label'}
    )
    # fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_test'], mode='lines', name='Predicted discharge')

    fig.update_layout(
        title={
            'font_size': 24,
            'xanchor': 'center',
            'x': 0.5
        }
    )

    chart = fig.to_html()
    context = {'chart': chart, 'form': DateForm()}  # Assuming DateForm is passed to the context
    return render(request, 'plotly_train.html', context)

def select_view(request):
    df = pd.read_csv('csv_files/metric.csv')
    context = {'df': df}
    return render(request, 'model_op.html', context)


import pandas as pd
import plotly.express as px

def plotly_scatter(request):
    # Load data
    df = pd.read_csv('csv_files/train.csv')

    # Get start and end dates from request parameters
    start = request.GET.get('start')
    end = request.GET.get('end')

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter DataFrame based on date range
    if start and end:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        filtered_df = df

    # Create scatter plot
    fig = px.scatter(
        filtered_df,
        x='y_train',
        y='pred_train',
        title="Scatter Plot",
        labels={'y_train': 'Observed', 'pred_train': 'Predicted'}
    )

    # Get maximum value from both x and y axes
    max_value = max(filtered_df[['y_train', 'pred_train']].max())

    # Set same scale on both axes
    fig.update_xaxes(range=[0, max_value], scaleratio=1)
    fig.update_yaxes(range=[0, max_value], scaleratio=1)

    # Update layout
    fig.update_layout(
        title={'font_size': 24, 'xanchor': 'center', 'x': 0.5}
    )

    # Convert plot to HTML
    chart = fig.to_html()

    # Assuming DateForm is passed to the context
    context = {'chart': chart, 'form': DateForm()}
    return render(request, 'plotly_train.html', context)


