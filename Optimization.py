import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
import random
import math
from random import shuffle, choice, uniform, randint
from HomeworkFramework import Function



class JADE(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.pop_size = 10
        self.population = []
        self.bounds = [i for i in zip(np.full(self.dim, self.lower), np.full(self.dim, self.upper))]
        self.p = max(0.05, 2/self.pop_size)
        self.c = 0.1

    def gen(self):
        for i in range(self.pop_size):
            pop_i = []
            for j in range(self.dim):
                pop_i.append(uniform(self.bounds[j][0], self.bounds[j][1]))
            self.population.append(pop_i)
    
    def eval(self):
        f_pop = []
        for pop_i in self.population:
            f_pop.append(self.f.evaluate(func_num, pop_i))
        return f_pop

    
    def bound(self, u_i_g):
        for j in range(self.dim):
            lower, upper = self.bounds[j]
            if u_i_g[j] < lower:
                u_i_g[j] = min(upper, 2*lower-u_i_g[j])
            if u_i_g[j] > upper:
                u_i_g[j] = max(lower, 2*upper-u_i_g[j])
        # return u_i_g

    def get_optimal(self, f_pop):
        f_best = f_pop[0]
        best = [val for val in self.population[0]]
        for i in range(1, self.pop_size):
            if f_pop[i] < f_best:
                f_best = float(f_pop[i])
                best = [val for val in self.population[i]]
        return f_best, best

    def mutation(self, i, best, f, Cr):
        idx = [x for x in range(self.pop_size) if x != i]
        r1_idx = np.random.choice(idx)
        r1 = self.population[r1_idx]
        idx.remove(r1_idx)

        r2 = self.population[np.random.choice(idx)]

        pop_i = self.population[i]
        u_i_g = []

        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(pop_i[j] + f*(best[j]-pop_i[j]) + f*(r1[j] - r2[j]))
            else:
                u_i_g.append(pop_i[j])

        return u_i_g
    
    def lehmer(self, f):
        return sum(x*x for x in f) / float(sum(f))

    def run(self, FES):
        self.gen()
        f_pop = self.eval()
        self.eval_times = self.pop_size
        u_cr, u_f = 0.5, 0.5
        scr = []
        sf = []

        while self.eval_times < FES:
            # print('=====================FE=====================')
            # print(self.eval_times)
            for i in range(self.pop_size):
                while True:
                    f = min(u_f + 0.1 * np.random.standard_cauchy(), 1)
                    if f > 0:
                        break
                #truncate to [0,1]
                cr = sorted((0, np.random.normal(u_cr, 0.1), 1))[1]

                #get p best
                best_index = np.argsort(f_pop)[:max(2, int(np.ceil(self.p*self.pop_size)))]
                p_idx = np.random.choice(best_index)
                u_i_g = self.mutation(i, self.population[p_idx], f, cr)
                self.bound(u_i_g)
                fitness = self.f.evaluate(func_num, u_i_g)
                if fitness == "ReachFunctionLimit":
                    self.eval_times += 1
                    print("ReachFunctionLimit")
                    break
                if fitness < f_pop[i]:
                    f_pop[i] = fitness
                    self.population[i] = u_i_g
                    scr.append(cr)
                    sf.append(f)

                self.eval_times += 1
            u_cr = (1-self.c) * u_cr + self.c * np.mean(scr)
            u_f = (1-self.c) * u_f + self.c * self.lehmer(sf)
            f_best, best = self.get_optimal(f_pop)
            # print("optimal_value = {}".format(f_best))

        return best, f_best

class CODE(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0

        self.param = [[1.0,0.1],[1.0,0.9],[0.8,0.2]]
        self.pop_size = 10
        self.bounds = [i for i in zip(np.full(self.dim, self.lower), np.full(self.dim, self.upper))]
        self.population = []

    def gen(self):
        for i in range(self.pop_size):
            pop_i = []
            for j in range(self.dim):
                pop_i.append(uniform(self.bounds[j][0], self.bounds[j][1]))
            self.population.append(pop_i)
    
    def eval(self):
        f_pop = []
        for pop_i in self.population:
            f_pop.append(self.f.evaluate(func_num, pop_i))
        return f_pop


    def rand_1(self,f, pop_i, Cr):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        r3 = pop_i
        while r3 == pop_i or r3 == r2 or r3 == r1:
            r3 = choice(self.population)

        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(r1[j] + f*(r2[j]-r3[j]))
            else:
                u_i_g.append(pop_i[j])

        return u_i_g

    def rand_2(self, f, pop_i, Cr):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        r3 = pop_i
        while r3 == pop_i or r3 == r2 or r3 == r1:
            r3 = choice(self.population)
        
        r4 = pop_i
        while r4 == pop_i or r4 == r1 or r4 == r2 or r4 == r3:
            r4 = choice(self.population)
        
        r5 = pop_i
        while r5 == pop_i or r5 == r2 or r5 == r1 or r5 == r3 or r5 == r4:
            r5 = choice(self.population)

        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(r1[j] + f*(r2[j]-r3[j])+ f*(r4[j]-r5[j]))
            else:
                u_i_g.append(pop_i[j])

        return u_i_g

    def current_to_rand_1(self, f, pop_i, Cr):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        r3 = pop_i
        while r3 == pop_i or r3 == r2 or r3 == r1:
            r3 = choice(self.population)
        
        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            rand = uniform(0,1)
            u_i_g.append(pop_i[j] + rand*(r1[j]-pop_i[j])+ f*(r2[j]-r3[j]))

        return u_i_g

    def current_to_best_1(self, f, pop_i, Cr, best):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(pop_i[j] + f*(best[j]-pop_i[j])+ f*(r1[j]-r2[j]))
            else:
                u_i_g.append(pop_i[j])
        return u_i_g

    def best_1(self, f, pop_i, Cr, best):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(best[j] + f*(r1[j]-r2[j]))
            else:
                u_i_g.append(pop_i[j])
        return u_i_g

    def best_2(self, f, pop_i, Cr, best):
        r1 = pop_i
        while r1 == pop_i:
            r1 = choice(self.population)
        
        r2 = pop_i
        while r2 == pop_i or r2 == r1:
            r2 = choice(self.population)
        
        r3 = pop_i
        while r3 == pop_i or r3 == r2 or r3 == r1:
            r3 = choice(self.population)
        
        r4 = pop_i
        while r4 == pop_i or r4 == r1 or r4 == r2 or r4 == r3:
            r4 = choice(self.population)

        u_i_g = []
        for j in range(self.dim):
            j_rand = randint(0,self.dim-1)
            if uniform(0,1) <= Cr or j == j_rand:
                u_i_g.append(best[j] + f*(r1[j]-r2[j])+ f*(r3[j]-r4[j]))
            else:
                u_i_g.append(pop_i[j])

        return u_i_g
    
    def bound(self, u_i_g):
        for j in range(self.dim):
            lower, upper = self.bounds[j]
            if u_i_g[j] < lower:
                u_i_g[j] = min(upper, 2*lower-u_i_g[j])
            if u_i_g[j] > upper:
                u_i_g[j] = max(lower, 2*upper-u_i_g[j])
        return u_i_g

    def get_optimal(self, f_pop):
        f_best = f_pop[0]
        best = [val for val in self.population[0]]
        for i in range(1, self.pop_size):
            if f_pop[i] < f_best:
                f_best = float(f_pop[i])
                best = [val for val in self.population[i]]
        return f_best, best

    def check(self, val):
        values = self.f.evaluate(func_num, val)
        if values == "ReachFunctionLimit":
            return None
        else:
            return values

    def run(self, FES):
        self.gen()
        f_pop = self.eval()
        self.eval_times = self.pop_size

        f_best, best = self.get_optimal(f_pop)

        while self.eval_times < FES:
            # print('=====================FE=====================')
            # print(self.eval_times)
            for i in range(self.pop_size):
                # print('=====================OI=====================')
                # print(self.eval_times)
                if self.eval_times + 3 >= FES:
                    break
                shuffle(self.param)
                candidate = []
                f_cand = []
                candidate.append(self.rand_1(self.param[0][0], self.population[i], self.param[0][1]))
                self.bound(candidate[0])

                # candidate.append(self.best_2(self.param[0][0], self.population[i], self.param[0][1], best))
                # self.bound(candidate[0])

                # shuffle(self.param)
                # candidate.append(self.rand_2(self.param[1][0], self.population[i], self.param[1][1]))
                # self.bound(candidate[1])

                shuffle(self.param)
                candidate.append(self.best_1(self.param[1][0], self.population[i], self.param[1][1], best))
                self.bound(candidate[1])

                # shuffle(self.param)
                # candidate.append(self.current_to_rand_1(self.param[2][0], self.population[i], self.param[2][1]))
                # self.bound(candidate[2])
                shuffle(self.param)
                candidate.append(self.current_to_best_1(self.param[2][0], self.population[i], self.param[2][1], best))
                self.bound(candidate[2])

                values = self.check(candidate[0])
                if values is not None: 
                    f_cand.append(values)
                else:
                    break
                
                values = self.check(candidate[1])
                if values is not None: 
                    f_cand.append(values)
                else:
                    break
                
                values = self.check(candidate[2])
                if values is not None: 
                    f_cand.append(values)
                else:
                    break

                index = np.argmin(f_cand)
                if f_cand[index] < f_pop[i]:
                    f_pop[i] = f_cand[index]
                    self.population[i] = candidate[index]
                
                self.eval_times+=3
            f_best, best = self.get_optimal(f_pop)
            print("optimal_value = {}".format(f_best))
            if self.eval_times + 3 >= FES:
                    break
        return best, f_best
                
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

        # you should implement your optimizer
        op = JADE(func_num)
        best_input, best_value = op.run(fes)

        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
