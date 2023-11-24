import numpy as np
from gurobipy import *
from gurobipy import Model
from gurobipy import GRB
import math

class MIP():
    def __init__(self, env, demand, sell_price, price, quantity, transport_time):
        self.env = env
        self.demand = demand
        self.sell_price = sell_price
        self.price = price
        self.transport_time = transport_time
        self.quantity = quantity
        self.perishability = self.env.perishability
        self.vehicle_capacity = self.env.vehicle_capacity
        self.products = range(self.env.num_products)
        self.suppliers = range(self.env.num_suppliers)
        self.periods = range(self.env.num_periods)

    def run_model(self):
        m = Model("Modelo Proyecto")
        I = m.addVars(range(self.env.num_products), range(self.env.num_periods), vtype=GRB.CONTINUOUS, name="I")
        X = m.addVars(range(self.env.num_suppliers), range(self.env.num_periods), vtype=GRB.INTEGER, name="X")
        Z = m.addVars(range(self.env.num_suppliers), range(self.env.num_products), range(self.env.num_periods), vtype=GRB.CONTINUOUS, name="Z")

        m.setObjective(quicksum(quicksum(self.demand[t, p]*self.sell_price[p] for p in self.products)/100 - quicksum(self.price[t,s,p] * Z[s,p,t] for p in self.products for s in self.suppliers)/100 - self.env.unit_transport_cost * quicksum(X[s, t]*self.transport_time[s] for s in self.suppliers) for t in self.periods), GRB.MAXIMIZE)
        for t in self.periods:
            for p in self.products:
                if t > 0:
                    m.addConstr(I[p, t] == I[p, t - 1]*(1-self.perishability) + quicksum(Z[s, p, t] for s in self.suppliers) - self.demand[t, p])
                else:
                    m.addConstr(I[p, t] == 0)
            for s in self.suppliers:
                m.addConstr(quicksum(Z[s, p, t] for p in self.products)/self.vehicle_capacity <= X[s, t])

                for p in self.products:
                    m.addConstr(Z[s, p, t] <= self.quantity[t, s, p])

        print(m.optimize())
