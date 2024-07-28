import numpy as np
class Inventory():
    def __init__(self, inv_capacity, actions, prob_demand, holding_cost, production_cost, profit):
        self.S = inv_capacity
        self.demand= prob_demand
        self.A = actions
        self.inv = holding_cost
        self.prod = production_cost
        self.gain = profit
    

    def Cost_vector(self):
        Cost=[]
        for i in range(self.A):
            C = []
            for j in range(self.S + 1):
                prod_cost = self.prod*i
                selling_profit=0
                holding_cost=0
                for iterate, prob in enumerate(self.demand[1:]):
                    if(iterate + 1 > j + i):
                      selling_profit += prob*self.gain*(i+j)
                    else:
                        selling_profit += prob*self.gain*(iterate+1)
                        holding_cost += prob*(i+j-iterate-1)*self.inv
                C.append(prod_cost+holding_cost-selling_profit)
            Cost.append(C)
        return np.array(Cost)
    
    def Prob(self):
        Probab=[]
        for i in range(self.A):
            P_state=np.zeros((self.S + 1, self.S + 1))
            for j in range(self.S + 1):
                for iterate, prob in enumerate(self.demand):
                    if(j+i-iterate>self.S):
                        P_state[j,self.S]+=prob

                    elif(j+i-iterate < 0):
                        P_state[j,0]+=prob

                    else:
                        P_state[j,j+i-iterate]+=prob
            Probab.append(P_state)
        return Probab
    
    
inv= Inventory(inv_capacity=3, actions=4, prob_demand=[0,0.5,0.5], holding_cost=0.5, production_cost=1, profit=2)
print(inv.Cost_vector())