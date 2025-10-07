import gurobipy as gp
from gurobipy import GRB
import numpy as np


def EVPL_agent_profit(offer_energy_price,offer_feed_in_price):
    # Parameters
    T = 24
    N = 3 # EV number
    delta_t = (24/T)
    eta_ch_ESS = 0.9
    eta_dis_ESS = 0.9
    P_max_ESS_ch = 10
    P_max_ESS_dis = 10
    ESS_cap = 13.5*3
    E_min_ESS = ESS_cap * 0.1
    E_max_ESS = ESS_cap * 0.9
    E0_ESS = 10


    eta_ch_EV = 0.9
    eta_dis_EV = 0.9
    P_max_EV_ch = 10
    P_max_EV_dis = 10
    EV_cap = 30
    E_min_EV =EV_cap * 0.1
    E_max_EV =EV_cap * 0.9

    # arrival and departure time (test)
    t_a = [3,10,11]
    t_d = [20,19,15]

    # # market price(test)
    # market_prices = np.random.rand(T)
    # feed_in_prices = 0.9 * market_prices
    # Grid_energy_price = np.random.rand(T)
    Grid_energy_price =np.array([0.55, 0.84, 0.32, 0.45, 0.30, 0.90, 0.75, 0.40, 0.12, 0.34, 0.75, 0.90,
                                  0.35, 0.33, 0.87, 0.91, 0.31, 0.50, 0.15, 0.86, 0.50, 0.31, 0.71, 0.30])



    # Model
    m = gp.Model("EVPL")

    # Add variables(RES)
    solar_irr=[0,0,0,0,0,0,0.07948244,0.240295749,0.458410351,0.744916821,0.922365989,0.981515712,0.914972274,0.737523105,0.593345656,0.452865065,0.258780037,0.153419593,0,0,0,0,0,0]
    epv = 0.16
    spv= 40
    P_PV={}
    for t in range(T):
        P_PV[t] = solar_irr[t] * epv * spv


    # EV
    # Add variables(EV)
    P_EV_ch = {}
    P_EV_dis = {}
    E_EV = {}
    charge_state_EV = {}

    for i in range(N):
        for t in range(T):
            P_EV_ch[i,t]= m.addVar(vtype=GRB.CONTINUOUS, name=f"P_EV_ch{i}_{t}")
            P_EV_dis[i,t]= m.addVar(vtype=GRB.CONTINUOUS, name=f"P_EV_dis_{i}_{t}")
            E_EV[i,t]= m.addVar(lb=0, ub=E_max_EV, vtype=GRB.CONTINUOUS, name=f"E_EV_{i}_{t}")
            charge_state_EV[i, t] = m.addVar(vtype=GRB.BINARY, name=f"charge_state_EV_{i}_{t}")
            # m.addConstr(E_EV[i, t] >= E_min_EV, name=f"E_EV_min_bound_{i}_{t}")

    for i in range(N):
        for t in range(t_a[i],t_d[i]):
            E_EV[i, t] = m.addVar(lb=E_min_EV, ub=E_max_EV, vtype=GRB.CONTINUOUS, name=f"E_EV_{i}_{t}")

    # set EV energy
    E_EV_ini = [12,16,4]
    E_EV_max = {}
    for i in range(N):
        potential_max_energy = E_EV_ini[i] + eta_ch_EV * P_max_EV_ch * (t_d[i] - t_a[i])
        E_EV_max[i] = min(potential_max_energy, E_max_EV)  # Set the upper bound



  
    return optimal_values
