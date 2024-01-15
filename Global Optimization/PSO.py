import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function
import random
# from smt.surrogate_models.surrogate_model import SurrogateModel

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

        # Particle Swarm Optimization實作
        self.pop_size=10
        self.solutions=[]
        self.individual_best_solution=[]
        self.individual_objective_value=[]

        # C1、C2、W
        self.cognition_factor=2
        self.social_factor=2
        self.reach=0
    def get_optimal(self):
        return self.optimal_solution,self.optimal_value 
    def initialize(self):
        min_index=0
        min_val=float("inf")
        for i in range(self.pop_size):
            solution=np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            self.solutions.append(solution)
            self.individual_best_solution.append(solution)
            objective=self.f.evaluate(func_num, solution)
            self.eval_times += 1
            self.individual_objective_value.append(objective)

            if objective<min_val:
                min_index=i
                min_val=objective
        self.optimal_solution=self.solutions[min_index].copy()
        self.optimal_value =min_val
    def move(self):
        for i,solution in enumerate(self.solutions):
            C1=self.cognition_factor*random.random()
            C2=self.social_factor*random.random()
            for d in range(self.dim):
                v=C1*(self.individual_best_solution[i][d]-self.solutions[i][d])+C2*(self.optimal_solution[d]-self.solutions[i][d])
                self.solutions[i][d]=0.7*self.solutions[i][d]+v
                self.solutions[i][d]=min(self.solutions[i][d],self.upper)
                self.solutions[i][d]=max(self.solutions[i][d],self.lower)
    def update_solution(self):
        for i,solution in enumerate(self.solutions):
            obj_val=self.f.evaluate(func_num,solution)
            self.eval_times += 1
            if obj_val == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                self.reach=1
                break            
            if(obj_val<self.individual_objective_value[i]):
                self.individual_best_solution[i]=solution
                self.individual_objective_value[i]=obj_val
                if obj_val<self.optimal_value :
                    self.optimal_solution=solution
                    self.optimal_value =obj_val
    def run(self, FES): # main part for your implementation
        self.initialize()
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            self.move()
            self.update_solution()

            
            print("optimal: {}\n".format(self.optimal_value ))
            if self.reach==1:
                break
            

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
