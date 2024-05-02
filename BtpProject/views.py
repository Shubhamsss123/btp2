from django.shortcuts import render,HttpResponse 
import plotly.express as px
from datetime import timedelta,datetime
from .forms import DateForm

import pandas as pd
import numpy as np 
import hydroeval as hy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error

from hyperopt import hp, tpe, Trials, fmin
from tensorflow.keras.optimizers import Adam

def temp(request):

    return render(request,'model_op.html',{'r2':1})
def home_page(request):

    return render(request,'home.html')

def model_selection_view(request):

    return render(request,'model_select.html')

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
        target = 'Discharge'

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
    fig=px.line(df,x='Date',y='Discharge')



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


    col1={"R2": "R2",
        "MAPE": "MAPE",
        "MSE": "MSE",
        "NSE":"NSE",
        "KGE":"KGE"}
    df = pd.DataFrame([col1,metrics_train_rounded, metrics_test_rounded]).T
    # Create the DataFrame with transposed metrics
    # df = pd.DataFrame([col1,metrics_train, metrics_test]).T
    df.index.name = "Metric"  # Set index name
    df.columns = ["metrics ","Train", "Test"]  # Set column names

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
    best = fmin(fn=lstm_hyperopt, space=lstm_space, algo=tpe.suggest, max_evals=1, trials=trials)

    best_params = {
        'n_steps': [3, 5, 7][best['n_steps']],
        'n_features': 1,
        'n_units': [50, 100][best['n_units']],
        'activation': ['relu', 'tanh'][best['activation']],
        'learning_rate': best['learning_rate']
    }
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
        # 'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
        'learning_rate': hp.uniform('learning_rate', 0.001),
        'n_steps':  hp.choice('n_steps', [3, 5, 7]),
        'n_features': 1,

    }

    # Run Bayesian Optimization
    trials = Trials()
    best = fmin(fn=ann_objective, space=ann_space, algo=tpe.suggest, max_evals=1, trials=trials)

    # print("Best hyperparameters:", best)

    # Use the best hyperparameters to build the final ANN model
    best_params = {
        'n_units': [8,16,32][best['n_units']],
        'learning_rate': best['learning_rate'],
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
    rf_best = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=1, trials=rf_trials)
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
    svm_best = fmin(fn=svm_objective, space=svm_space, algo=tpe.suggest, max_evals=1, trials=svm_trials)

    # print("Best SVM hyperparameters:", svm_best)

    # Use the best hyperparameters to build the final SVM model
    svm_best_params = {
        'C': svm_best['C'],
        'kernel': ['linear', 'rbf', 'poly'][svm_best['kernel']],
        'n_steps': [3, 5, 7][svm_best['n_steps']],
        'n_features': 1,
    }

    return svm_best_params

def data_view(request):
    
    if request.method == 'POST':
        # Collect form data
        ml_task = request.POST.get('ml_task')  # Get the selected regression task
        train_split = request.POST.get('train_split')  # Get the selected train split
        seasonality = request.POST.get('season')  # Check if seasonality checkbox is checked
        csv_file = request.FILES.get('csv_name')  # Get the uploaded CSV file

        print(ml_task,train_split,seasonality,csv_file)
    
        if csv_file:
            

            print(ml_task,train_split,seasonality,csv_file)

        
            try:
                df=pd.read_csv(csv_file)
            except:
                df=pd.read_excel(csv_file)

            df.columns=['date','discharge']
            df.set_index('date',drop=True,inplace=True)

            dates=df.index
            print(dates[-1],type(dates[-1]))
            try:
                future_30dates=[datetime.strptime(dates[-1], '%Y-%m-%d') + timedelta(days=i) for i in range(1, 31)]
            except:
                future_30dates=[dates[-1] + timedelta(days=i) for i in range(1, 31)]

            train_len=int(df.shape[0]*int(train_split)/100)
        

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
                    C=svm_best_params['C']
                    kernel=svm_best_params['kernel']
                    n_steps = best_params['n_steps']
                    
                    X, y = split_sequence(raw_seq, n_steps)
                    X = X.reshape((X.shape[0], X.shape[1], n_features))
                    X_train,X_test,y_train,y_test = X[:train_len, :],X[train_len:, :],y[:train_len],y[train_len:]
                    model = SVR(C=C, kernel=kernel)
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

            for i in range(30):
                if ml_task == 'ann' or ml_task == 'lstm':
                    y_pred = model.predict(last_n.reshape(1, n_steps, 1)).flatten()[0]
                else:
                    y_pred = model.predict(last_n.reshape(1, n_steps)).flatten()[0]
                
                future_pred.append(y_pred)
                last_n = np.append(last_n[1:], y_pred)

          
                

            df_pred=pd.DataFrame({'date':future_30dates,'pred':future_pred})
            df_pred.to_csv(f'csv_files/pred.csv',index=None)

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
            return render(request,'model_op.html',{'df':df1})
            # return HttpResponse('csv files saved')
        else:
            return HttpResponse('csv file is not uploaded ')

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
    fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_train'], mode='lines', name='Predicted Discharge')

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
    fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_test'], mode='lines', name='Predicted Discharge')

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
    # fig.add_scatter(x=filtered_df['date'], y=filtered_df['pred_test'], mode='lines', name='Predicted Discharge')

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