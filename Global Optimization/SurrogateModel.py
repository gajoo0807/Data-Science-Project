# import all packages and set plots to be embedded inline
import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function
import sys
import random
import scipy.stats as st

from scipy.integrate import quad
import sklearn.gaussian_process as gp
import seaborn as sns
import statsmodels.api as sm 
import pandas as pd
import os

class Surrogate_model(Function):
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf") # 正無限大實體物件
        self.optimal_solution = np.empty(self.dim)
        self.reach=0
    # Expected Improvement-based active learning function
    def EI_learning(candidates, y_pred, pred_std):
        """Active learning function based on expected improvement
       
            This function selects a new sample from candidate pool to enrich the current training dataset.
            The sample gets selected if it has the maximum expected improvement value.
       
            Input: 
            - candidates: pool of candidates to select sample from
            - y_pred: GP predictions on candidate samples
            - pred_std: Standard deviation of the GP predictions on candidate samples
       
             Output:
            - new_sample: the selected sample with the maximum expected improvement value
            - EI: EI values of the candidate samples
        """  
        # 1-Find the current minimum
        current_objective = y_pred[np.argmin(y_pred)]
    
        # 2-Calculate the EI values of the candidate samples
        pred_std = pred_std.reshape(pred_std.shape[0], 1)
        EI = (current_objective-y_pred)*st.norm.cdf((current_objective-y_pred)/pred_std) \
                +pred_std*st.norm.pdf((current_objective-y_pred)/pred_std)
    
        # 3-Select a new sample
        new_sample = candidates[np.argmax(EI)]
        return new_sample, EI
    def init_build(self):
        self.X_train=[]
        self.y_train=[]
        self.candidates=[]
        for i in range(100):
            solution=np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            objective=self.f.evaluate(func_num,solution)  
            self.eval_times += 1
            self.X_train.append(solution)
            self.y_train.append(objective)

        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-3, 1e3))
        self.model = gp.GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',n_restarts_optimizer=30, alpha=1e-10,normalize_y=True)   
        self.model.fit(X_train,y_train) 
    def calculate(self):
        # 1 GP predicting
        for i in range(100):
            solution=np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            X_test.append(solution)
        y_pred,pred_std=self.model.predict(X_test.reshape(-1,1),return_std=True)
        pred_std = pred_std.reshape(pred_std.shape[0], 1)
        index=np.argmin(y_pred)

        # 2-Calculate the current minimum
        current_min = y_pred[np.argmin(y_pred),0] # argmin :最小值對應的索引
        location = candidates[np.argmin(y_pred)]
        # current_location=X_test[index]
        # current_objective = y_pred[index]
        
        # 3-Select next sample
        pred_std += 1e-8        # To prevent zero standard deviation value
        new_sample, EI = EI_learning(candidates, y_pred, pred_std) 

        # 4-Calculate the true label of the new sample
        y_sample=[]
        for sol in new_sample:
            obj=self.f.evaluate(sol)
            if objective == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                self.reach=1
                break  
            self.eval_times += 1
            y_sample.append(obj)
        X_train = np.vstack(new_sample) # 沿着竖直方向将矩阵堆叠起来
        y_train = np.vstack(y_sample)
        iteration += 1
    def run(self,FES):
        self.init_build()
        while self.eval_times < FES:
            print('=====================FE=====================') 
            print(self.eval_times)
            self.calculate()
            if FES%100==0:
                   self.model.fit(X_train,y_train)
            
