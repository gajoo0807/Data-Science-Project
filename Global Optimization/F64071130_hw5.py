import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function
# from smt.surrogate_models.surrogate_model import SurrogateModel

# class RBF(SurrogateModel):
#     def _initialize(self):
#         super(RBF,self)._initialize()
#         ...
#     def _setup(self):
#         options=self.options
#     def _train(self):
#         self._setup()
#     def _predict_values(self,x):
#         """
#         Evaluates the model at a set of points,
#         x:np.array，input values
#         y:output,Evaluation point output values
#         """
#         n=x.shape[0]
#     def _predict_derivatives(self,x,kx):
#         """
#         Evaluates the derivatives at a set of points
#         x:np.array，evaluation point input variable values
#         kx:int
#         the 0-based index of the input variable with respect to which derivatives are desired
#         dy_dx:derivate values(output)
#         """
#         n=x.shape[0]
#     def _predict_output_derivatives(self,x):
#         """
#         Evaluates the output derivates at a set of points
#         x:np.array Evaluation point input variable values
#         dy_dyt:output derivative values
#         """
#         n=x.shape[0]
class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func
        
        self.eval_times = 0
        self.optimal_value = float("inf") # 正無限大實體物件
        self.optimal_solution = np.empty(self.dim)
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        
        while self.eval_times < FES:
            print('self.dim',self.dim)
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim) #在upper、lower間產生一浮點數

            value = self.f.evaluate(func_num, solution) #丟參數進去
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if float(value) < self.optimal_value: # find more optimal solution
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))
            

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500
        print(func_num)
        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
