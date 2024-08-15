#勾配木で予測する
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoost
from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from horse_simu import Simulator


class my_model:
  def __init__(self,name,x_data,y_data,test_x,test_y):
    if name == 'lgb':
      self.model_name = 'lgb'
    elif name == 'xgb':
      self.model_name = 'xgb'
    elif name == 'cb':
      self.model_name = 'cb'

    self.x_data = x_data
    self.y_data = y_data
    self.test_x = test_x
    self.test_y = test_y
    self.results = {}
    #self.scores = []
    self.history = []
    self.simu = Simulator('race_data.csv')

  def accuracy(self,pred,y):
    print(y)
    print(pred)
    return confusion_matrix(y,pred)

  def train_model(self,params):

    n_splits = 7
    kf = StratifiedKFold(n_splits = n_splits,shuffle = True,random_state = 73)
    scores =[]
    self.predict_test = 0
    for tr_idx,va_idx in kf.split(self.x_data,self.y_data):
      tr_x,va_x = self.x_data.iloc[tr_idx],self.x_data.iloc[va_idx]
      tr_y,va_y = self.y_data.iloc[tr_idx],self.y_data.iloc[va_idx]

      print(tr_x.shape,va_x.shape,tr_y.shape,va_y.shape)


      if self.model_name == 'lgb':
        params['n_estimators'] = int(params['n_estimators'])
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        lgb_train = lgb.Dataset(tr_x,tr_y)
        lgb_val = lgb.Dataset(va_x,va_y)
        self.model = lgb.train(params,lgb_train,num_boost_round = 10000,valid_names = ['train','valid'],
                               #categorical_feature=['Sex'],
                    valid_sets = [lgb_train,lgb_val],callbacks=[lgb.record_evaluation(self.results),lgb.early_stopping(stopping_rounds=10,
                                verbose=True)])
        val_pred = self.model.predict(va_x)
        #self.predict_test += self.model.predict(self.test_data.drop(['id'],axis = 1)) / n_splits
        #val_pred = np.rint(val_pred)
        #score = my_rmsle(va_y,val_pred)
        #scores.append(score)

        #print(score)
      elif self.model_name == 'xgb':
        params['max_depth'] = int(params['max_depth'])
        xgb_train = xgb.DMatrix(tr_x,label = tr_y)
        xgb_val = xgb.DMatrix(va_x,label = va_y)
        xgb_test = xgb.DMatrix(self.test_data.drop(['id'],axis = 1))
        watchlist = [(xgb_train,'train'),(xgb_val,'valid')]
        self.model = xgb.train(params,
                               xgb_train,
                               num_boost_round = 5000,
                               evals = watchlist,
                               evals_result = self.results,
                               early_stopping_rounds = 10,
                               verbose_eval = 200
                               )

        self.predict_test += self.model.predict(xgb_test) / n_splits
        val_pred = self.model.predict(xgb_val)
        score = my_rmsle(va_y,val_pred)
        scores.append(score)
      elif self.model_name == 'cb':
        #params['max_depth'] = int(params['max_depth'])
        cb_train = Pool(tr_x,tr_y,cat_features = ['Sex'])
        cb_val = Pool(va_x,va_y,cat_features = ['Sex'])
        cb_test = Pool(self.test_data.drop(['id'],axis = 1),cat_features = ['Sex'])
        watchlist = [(cb_train,'train'),(cb_val,'valid')]
        self.model = CatBoost(params)
        self.model.fit(cb_train,
                       early_stopping_rounds=10
                       )


        self.predict_test += self.model.predict(cb_test) / n_splits
        val_pred = self.model.predict(cb_val)
        score = my_rmsle(va_y,val_pred)
        scores.append(score)

    #print(np.mean(scores))

      #結果の出力
      #predict = self.model.predict(test_df.drop(['id'],axis = 1))

      #pred = np.rint(pred)
    """
    submission = pd.DataFrame({"id" : test_df["id"],"Rings": predict})
    submission.head()
    submission.to_csv("/content/submission_abalone.csv", index=False)
    """
    #self.history.append((params,np.mean(scores)))
    #return np.mean(scores)
    #return {'loss' : np.mean(scores),'status' : STATUS_OK}
  def eval_predict(self,x,y):
    predict = self.predict(x)
    np.savetxt('temp.txt', predict)
    predict = np.argmax(predict,axis=1)
    return self.accuracy(predict,y)
  def predict(self,x):
    if self.model_name == 'lgb':
      predict = self.model.predict(x)
    elif self.model_name == 'xgb':
      predict = self.model.predict(xgb.DMatrix(x))
    elif self.model_name == 'cb':
      predict = self.model.predict(Pool(x,cat_features = ['Sex']))
    return predict
  def plot_loss(self):
    loss_train = self.results['train']['rmse']
    loss_test = self.results['valid']['rmse']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('rmse')

    ax1.plot(loss_train, label='train loss')
    ax1.plot(loss_test, label='test loss')

    plt.legend()
    plt.show()
  def get_hist(self):
    return self.history
  def save_model(self):
    file = '/content/' + self.model_name + '.pkl'
    pickle.dump(self.model, open(file, 'wb'))
  def get_predict_test(self):
    return self.predict_test
  def simulate(self,x,id_list):
    
    predicts = self.predict(x)
    predicts = np.argmax(predicts,axis = 1) + 1
    for i,predict in enumerate(predicts):
      id = id_list[i]
      print(id,predict)
      self.simu.buy_ticket2(str(id),0,predict)
    self.simu.print_info()

if __name__ == '__main__':
    #train_df = pd.read_csv('datas/train_data.csv')
    #test_df = pd.read_csv('datas/test_data.csv')
    df = pd.read_csv('datas/data_gbdt.csv')
    df['date'] = pd.to_datetime(df['date'])
    #ここでdfに対して処理をして特徴量を作成する
    train_df = df.loc[df['date'] <= np.datetime64('2024-07-01 00') ]
    test_df = df.loc[df['date'] > np.datetime64('2024-07-01 00') ]
    print(len(train_df),len(test_df))
    """
    分けた二つのデータ内では時系列順に並んでいないので混ぜちゃダメ
    df_all = pd.concat([train_df,test_df])
    train_df = df_all[:4650]
    test_df = df_all[4650:]

    print(len(train_df),len(test_df))
    """
    x_train = train_df.drop(['winner','ID','date'],axis=1)
    y_train = train_df['winner'].apply(lambda x: x-1)
    test_x = test_df.drop(['winner','ID','date'],axis=1)
    test_y = test_df['winner'].apply(lambda x: x -1)
    id_list = test_df['ID'].to_list()
    
    #オッズを消去する
    
    for i in range(18):
      for j in range(2):
        x_train = x_train.drop(['人気'+str(i+1)+'_'+str(j+1)],axis=1)
        test_x = test_x.drop(['人気'+str(i+1)+'_'+str(j+1)],axis=1)
    #print(x_train)
    

    param = {'reg_alpha': 2.641145545918664e-07,
         'boosting_type': 'gbdt',
         'metric': {'multi_error'},
         'objective': 'multiclass',
         'num_class':18,
         'seed': 38,
         'verbosity': -1,
         'reg_lambda': 7.327538824227066, 'max_depth': 14, 'num_leaves': 179, 'n_estimators': 943, 'subsample': 0.9920242437270281, 'learning_rate': 0.014039242387856176, 'colsample_bytree': 0.7195138436765769, 'subsample_for_bin': 261570}

    my_lgbpre = my_model('lgb',x_train,y_train,test_x,test_y)

    my_lgbpre.train_model(param)

    print(my_lgbpre.eval_predict(test_x,test_y))
    
    my_lgbpre.simulate(test_x,id_list)
