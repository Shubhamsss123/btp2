
import pandas as pd
import numpy as np 
import hydroeval as hy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error

from hyperopt import hp, tpe, Trials, fmin
from tensorflow.keras.optimizers import Adam

def evaluate_model_df(model, X_train, y_train, X_test, y_test):
  
    # Make predictions on training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    metrics_train = {
        "R2": r2_score(y_train, y_pred_train),
        "MAPE": mean_absolute_percentage_error(y_train, y_pred_train),
        "MSE": mean_squared_error(y_train, y_pred_train),
        "NSE": hy.nse(y_train, y_pred_train),
        # "KGE": hy.kge(y_train, y_pred_train)
    }

    
    metrics_test = {
        "R2": r2_score(y_test, y_pred_test),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "NSE": hy.nse(y_test, y_pred_test),
        # "KGE": hy.kge(y_test, y_pred_test)
    }
    # print('shape of y_pred_trian and test is ',y_pred_test.shape,y_pred_train.shape,sep=' => ')
    # print(metrics_train,'this is training metrics ')
    metrics_train_rounded = {key: round(value, 3) for key, value in metrics_train.items()}
    metrics_test_rounded = {key: round(value, 3) for key, value in metrics_test.items()}

    col1={"R2": "R2",
        "MAPE": "MAPE",
        "MSE": "MSE",
        "NSE":"NSE"}
    df = pd.DataFrame([col1,metrics_train_rounded, metrics_test_rounded]).T
    # Create the DataFrame with transposed metrics
    # df = pd.DataFrame([col1,metrics_train, metrics_test]).T
    df.index.name = "Metric"  # Set index name
    df.columns = ["metrics ","Train", "Test"]  # Set column names

    return df.round({'Train':2,'Test':2})

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




def train_model(ml_task,train_split,csv_file,seasonality): 

    #ml_task => 'ann' or 'lstm' or 'svm' or 'rf'

    if csv_file:
        

        print(ml_task,train_split,seasonality,csv_file)

       
        try:
            df=pd.read_csv(csv_file)
        except:
            df=pd.read_excel(csv_file)
        df.columns=['date','discharge']
        df.set_index('date',drop=True,inplace=True)

        dates=df.index

        l=df.shape[0]

        r=int(train_split)/100
        train=df['discharge'][:int(l*r)]
        test=df['discharge'][int(l*r):]
        dates=dates[3:]
        date_train=dates[:int(l*r)]
        date_test=dates[int(l*r):]

        if ml_task=='ann' or ml_task=='lstm':
            
            raw_seq=df['discharge'].values
            def lstm_hyperopt(params):
                n_steps = params['n_steps']
                n_features = params['n_features']
                n_units = params['n_units']
                activation = params['activation']
                learning_rate = params['learning_rate']

                # split into samples
                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                X_train = X[:-30, :]
                X_test = X[-30:, :]
                y_train = y[:-30]
                y_test = y[-30:]

                # define model
                model = Sequential()
                model.add(LSTM(n_units, activation=activation, input_shape=(n_steps, n_features)))
                model.add(Dense(1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')

                model.fit(X_train, y_train, epochs=1, verbose=0)

                yhat = model.predict(X_test, verbose=0)

                return -r2_score(y_test, yhat.flatten())
            
            n_steps = 3
            # split into samples
            X, y = split_sequence(raw_seq, n_steps)

            X_train=X[:int(l*r),:]
            X_test=X[int(l*r):,:]
            y_train=y[:int(l*r)]
            y_test=y[int(l*r):]
            n_features = 1


            if ml_task=='lstm':

                lstm_space = {
                    'n_steps': hp.choice('n_steps', [3, 5, 7]),
                    'n_features': 1,
                    'n_units': hp.choice('n_units', [50, 100]),
                    'activation': hp.choice('activation', ['relu', 'tanh']),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.01)
                }

                # Run Bayesian Optimization
                trials = Trials()
                best = fmin(fn=lstm_hyperopt, space=lstm_space, algo=tpe.suggest, max_evals=1, trials=trials)

                # model = Sequential()
                # model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
                # model.add(Dense(1))
                # model.compile(optimizer='adam', loss='mse')

                best_params = {
                    'n_steps': [3, 5, 7][best['n_steps']],
                    'n_features': 1,
                    'n_units': [50, 100][best['n_units']],
                    'activation': ['relu', 'tanh'][best['activation']],
                    'learning_rate': best['learning_rate']
                }

                n_steps = best_params['n_steps']
                n_features = best_params['n_features']
                n_units = best_params['n_units']
                activation = best_params['activation']
                learning_rate = best_params['learning_rate']

                # split into samples
                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                X_train = X[:-30, :]
                X_test = X[-30:, :]
                y_train = y[:-30]
                y_test = y[-30:]

                # define model
                model = Sequential()
                model.add(LSTM(n_units, activation=activation, input_shape=(n_steps, n_features)))
                model.add(Dense(1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')

                model.fit(X_train, y_train, epochs=10, verbose=1)

                # yhat = model.predict(X_test, verbose=1)                

            else:

                def ann_objective(params):
                    
                    n_units = params['n_units']
                    learning_rate = params['learning_rate']
                    n_steps= params['n_steps']


                    X, y = split_sequence(raw_seq, n_steps)
                    X = X.reshape((X.shape[0], X.shape[1], n_features))

                    X_train = X[:-30, :]
                    X_test = X[-30:, :]
                    y_train = y[:-30]
                    y_test = y[-30:]

                    

                    # Define and compile the ANN model
                    ann_model = Sequential()
                    ann_model.add(Dense(units=64, activation='relu', input_shape=(n_steps,)))
                    ann_model.add(Dense(units=n_units, activation='relu'))
                    ann_model.add(Dense(units=1))

                    optimizer = Adam(learning_rate=learning_rate)
                    ann_model.compile(optimizer=optimizer, loss='mse')

                    # Fit the model
                    ann_model.fit(X_train.reshape((X_train.shape[0], n_steps)), y_train, verbose=0, epochs=2)

                    # Predict on the test set and calculate R^2 score
                    yhat = ann_model.predict(X_test.reshape((X_test.shape[0], n_steps)), verbose=0)
                    r2 = r2_score(y_test, yhat.flatten())

                    return -r2  # Hyperopt minimizes the objective function, so negate R^2

                # Define the search space for Hyperopt
                ann_space = {
                    'n_units': hp.choice('n_units', [32, 64, 128]),
                    'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
                    'n_steps':  hp.choice('n_steps', [3, 5, 7])
                }

                # Run Bayesian Optimization
                trials = Trials()
                best = fmin(fn=ann_objective, space=ann_space, algo=tpe.suggest, max_evals=1, trials=trials)

                # print("Best hyperparameters:", best)

                # Use the best hyperparameters to build the final ANN model
                best_params = {
                    'n_units': [32, 64, 128][best['n_units']],
                    'learning_rate': best['learning_rate'],
                    'n_steps': [3, 5, 7][best['n_steps']],
                }                


                n_steps = best_params['n_steps']
            
                n_units = best_params['n_units']
              
                learning_rate = best_params['learning_rate']
                # model = Sequential()
                # model.add(Dense(units=64, activation='relu', input_shape=(3,))) 
                # model.add(Dense(units=32, activation='relu'))
                # model.add(Dense(units=1))  
                # model.compile(optimizer='adam', loss='mse')

                X, y = split_sequence(raw_seq, n_steps)
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                X_train = X[:int(l*r), :]
                X_test = X[int(l*r):, :]
                y_train = y[:int(l*r)]
                y_test = y[int(l*r):]

                model = Sequential()
                model.add(Dense(units=64, activation='relu', input_shape=(n_steps,)))
                model.add(Dense(units=n_units, activation='relu'))
                model.add(Dense(units=1))

                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')

                # Fit the final model
                model.fit(X_train.reshape((X_train.shape[0], best_params['n_steps'])), y_train, verbose=1, epochs=2)

            # model.fit(X_train,y_train,epochs=10,verbose=0)
            
            # pred=model.predict(X_test)

        else:

            
            raw_seq=df['discharge'].values
            n_steps = 3
            # split into samples
            X, y = split_sequence(raw_seq, n_steps)
            # df['last']=df['discharge'].shift(1)
            # df['second_last']=df['discharge'].shift(2)
            # df['third_last']=df['discharge'].shift(3)
            # df.dropna(inplace=True,axis=0)
            # x1,x2,x3,y=df['last'],df['second_last'],df['third_last'],df['discharge']
            # x1,x2,x3=np.array(x1),np.array(x2),np.array(x3)
            # X=np.column_stack((x1,x2,x3))


            X_train=X[:int(l*r),:]
            X_test=X[int(l*r):,:]
            y_train=y[:int(l*r)]
            y_test=y[int(l*r):]

        

            def rf_objective(params):
                n_estimators = params['n_estimators']
                max_features = params['max_features']

                rf_model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=1)
                rf_model.fit(X_train, y_train)
                score = rf_model.score(X_test, y_test)

                return -score  # Hyperopt minimizes the objective function, so negate score

            # Define objective function for SVM
            def svm_objective(params):
                C = params['C']
                kernel = params['kernel']

                svr = SVR(C=C, kernel=kernel)
                svr.fit(X_train, y_train)
                score = svr.score(X_test, y_test)

                return -score  # Hyperopt minimizes the objective function, so negate score

            #lag 1 to 10 
            rf_space = {
                'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
                'max_features': hp.choice('max_features', [2, 3, 4])
            }

            # Define the search space for SVM
            svm_space = {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly'])
            }
            if ml_task=='svm':


                svm_trials = Trials()
                svm_best = fmin(fn=svm_objective, space=svm_space, algo=tpe.suggest, max_evals=1, trials=svm_trials)

                # print("Best SVM hyperparameters:", svm_best)

                # Use the best hyperparameters to build the final SVM model
                svm_best_params = {
                    'C': svm_best['C'],
                    'kernel': ['linear', 'rbf', 'poly'][svm_best['kernel']]
                }

                model = SVR(C=svm_best_params['C'], kernel=svm_best_params['kernel'])
                model.fit(X_train, y_train)

                # model=SVR()


            
            else:

                rf_trials = Trials()
                rf_best = fmin(fn=rf_objective, space=rf_space, algo=tpe.suggest, max_evals=1, trials=rf_trials)

                # print("Best Random Forest hyperparameters:", rf_best)

                # Use the best hyperparameters to build the final Random Forest model
                rf_best_params = {
                    'n_estimators': [50, 100, 150][rf_best['n_estimators']],
                    'max_features': [2, 3, 4][rf_best['max_features']]
                }

                model = RandomForestRegressor(n_estimators=rf_best_params['n_estimators'],
                                                max_features=rf_best_params['max_features'],
                                                random_state=1)
                model.fit(X_train, y_train)

                # model=RandomForestRegressor(n_estimators=100,max_features=3,random_state=1)

            print(X_train.shape,y_train.shape,sep='====>')
            model.fit(X_train,y_train)

        model_dict={'lstm':" LSTM ",'ann':"Artificial Neural Network (ANN)",'rf':'Random Forest','svm':"Support Vector Machine"}
        pred_train=model.predict(X_train)
        pred=model.predict(X_test)

        print(f'shapes :train {pred_train.shape} test {pred.shape}  {date_train.shape} {date_test.shape}')

        df1=evaluate_model_df(model, X_train, y_train, X_test, y_test)
        
     
        # plt.figure(figsize=(5,7))
        # fig.autofmt_xdate()
        plt.plot(date_train[::300],y_train[::300],label='actual values')
        plt.plot(date_train[::300],pred_train[::300],c='r',label='prediction')
        plt.title(model_dict[ml_task]+" Train")
        plt.xlabel('per 300 days dates')
        plt.ylabel('discharge (m3/h)')
        plt.xticks(rotation=40)
        
        plt.tight_layout()

        plt.legend()
        plt.savefig('ml_train.png')
        plt.close()

        # plt.figure(figsize=(5,7))
        # fig.autofmt_xdate()
        plt.plot(date_test[::200],y_test[::200],label='actual values')
        plt.plot(date_test[::200],pred[::200],c='r',label='prediction')
       
        plt.title(model_dict[ml_task]+" Test")
        plt.xlabel('per 200 days dates')
        plt.ylabel('discharge (m3/h)')
        plt.xticks(rotation=40)
        plt.legend()
        plt.tight_layout()
        plt.savefig('ml_test.png')
        plt.close()

        print(df1)



        return 
