import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

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



    # EV charging/discharging, add constraints
    for i in range(N):
        for t in range(T):
            # Ensure EV energy is zero before arrival and after departure
            if t < t_a[i] or t > t_d[i]:
                m.addConstr(E_EV[i, t] == 0, name=f"E_EV_state_{i}_{t}")
                m.addConstr(P_EV_ch[i, t] == 0, name=f"P_EV_ch_state_{i}_{t}")  # Charging power is zero
                m.addConstr(P_EV_dis[i, t] == 0, name=f"P_EV_dis_state_{i}_{t}")  # Discharging power is zero
            elif t == t_a[i]:
                m.addConstr(E_EV[i, t] == E_EV_ini[i], name=f"E_EV_ini_state_{i}_{t}")
                m.addConstr(P_EV_ch[i, t] == 0, name=f"P_EV_ch_state_{i}_{t}")  # Charging power is zero
                m.addConstr(P_EV_dis[i, t] == 0, name=f"P_EV_dis_state_{i}_{t}")  # Discharging power is zero
            elif t_a[i] < t <= t_d[i]:
                m.addConstr(E_EV[i, t] == E_EV[i, t-1] + (eta_ch_EV * P_EV_ch[i, t] - (1/eta_dis_EV) * P_EV_dis[i, t]) * delta_t, name=f"E_EV_dynamic_{i}_{t}")
            elif t == t_d[i]:
                m.addConstr(E_EV[i, t] == E_EV_max[i], name=f"E_EV_max_state_{i}_{t}")

            m.addConstr(E_EV[i, t_d[i]] == E_EV_max[i], name=f"E_EV_departure_energy_{i}_{t}") # EV constraint ensure EV charging as much as possible
            m.addConstr(P_EV_ch[i, t] <= P_max_EV_ch * charge_state_EV[i, t])  # Charging constraint
            m.addConstr(P_EV_dis[i, t] <= P_max_EV_dis * (1 - charge_state_EV[i, t]))  # Discharging constraint

    # EV constraint ensure EV charging as much as possible
    # for i in range(N):
    #     net_energy_change = gp.quicksum(
    #         (eta_ch_EV * P_EV_ch[i, t] - (1 / eta_dis_EV) * P_EV_dis[i, t]) * delta_t
    #         for t in range(t_a[i] + 1, t_d[i] + 1)
    #     )
    #     m.addConstr(E_EV_max[i] == E_EV_ini[i] + net_energy_change, name=f"max_energy_constraint_{i}")


    # EV join V2G OR NOT

    phi = [0] * N

    for i in range(N):
        # Calculate the required time to reach maximum energy
        required_time = (E_EV_max[i] - E_EV_ini[i]) / (eta_ch_EV * P_max_EV_ch)

        # Determine the value of phi_i based on the condition
        if (t_d[i] - t_a[i]) > required_time:
            phi[i] = 1
        else:
            phi[i] = 0
            # m.getVarByName(f"P_EV_dis_{i}_{t}").lb = 0
            # m.getVarByName(f"P_EV_dis_{i}_{t}").ub = 0

    # Profit z_ev
    lambda_c=0.3 # may consider RL for pricing strategy
    lambda_inc=0.05

    ## Dynamic lambda_c

    lambda_c_max = max(offer_energy_price)+0.2
    lambda_c_min = min(offer_energy_price)+0.2

    lambda_c = m.addVars(N, name="lambda_c")

    E_i_de_max={}
    E_i_de={}
    for i in range(N):
        if phi[i] == 0:
            E_i_de_max[i]=EV_cap
            E_i_de[i]=E_EV_max[i]-E_EV_ini[i]
            m.addConstr(lambda_c[i] == lambda_c_max - (E_i_de[i] / E_i_de_max[i]) * lambda_c_max + (
                        E_i_de[i] / E_i_de_max[i]) * lambda_c_min, f"lambda_c_{i}")


    z_EV = gp.quicksum((1 - phi[i]) * gp.quicksum(P_EV_ch[i, t] * lambda_c[i] * delta_t for t in range(t_a[i], t_d[i])) for i in range(N)) + \
           gp.quicksum(phi[i] * ((E_EV_max[i] - E_EV_ini[i]) * lambda_c_min - gp.quicksum(P_EV_dis[i, t] * lambda_inc * delta_t for t in range(T))) for i in range(N))


    #ESS
    # Add variables(ESS)
    P_ESS_ch = m.addVars(T, lb=0, ub=P_max_ESS_ch, name="P_ESS_ch")
    P_ESS_dis = m.addVars(T, lb=0, ub=P_max_ESS_dis, name="P_ESS_dis")
    E_ESS = m.addVars(T, lb=E_min_ESS, ub=E_max_ESS, name="E_ESS")
    charge_state_ESS = m.addVars(T, vtype=GRB.BINARY, name="charge_state_ESS")  # Binary variable for charging state

    # Set initial energy
    E_ESS[0].lb = E0_ESS
    E_ESS[0].ub = E0_ESS

    # Add constraints
    for t in range(1, T):
        m.addConstr(E_ESS[t] == E_ESS[t-1] + (eta_ch_ESS * P_ESS_ch[t] - (1 / eta_dis_ESS) * P_ESS_dis[t]) * delta_t)
        m.addConstr(P_ESS_ch[t] <= P_max_ESS_ch * charge_state_ESS[t])      # Charging constraint
        m.addConstr(P_ESS_dis[t] <= P_max_ESS_dis * (1 - charge_state_ESS[t]))  # Discharging constraint

    # Ensure final energy equals initial energy
    m.addConstr(E_ESS[T-1] == E0_ESS)

    # battery degradation

    # Power balance
    # Define decision variable
    P_Grid = {}
    P_Feed_in = {}
    P_fvp = {}
    for t in range(T):
        P_Grid[t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"P_Grid_{t}")
        P_Feed_in[t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"P_Feed_in_{t}")
        P_fvp[t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"P_fvp_{t}")


    for t in range(T):
        m.addConstr(
            gp.quicksum(P_EV_ch[i, t] for i in range(N)) +
            P_ESS_ch[t] +
            P_Feed_in[t] ==
            P_PV[t] +
            P_Grid[t] +
            P_fvp[t] +
            P_ESS_dis[t] +
            gp.quicksum(P_EV_dis[i, t] for i in range(N)),
            "power_balance_t{}".format(t)
        )
    #will be use when adding FCAS
        # P_fvp[t]
        # gp.quicksum(P_l_disp[k, t] * tau_l[t] for k in range(N_l))
        # gp.quicksum(P_r_disp[j, t] * tau_r[t] for j in range(N_r))

    # Objective
    objective = gp.quicksum(P_Feed_in[t] * offer_feed_in_price[t] * delta_t for t in range(T)) - \
                gp.quicksum(P_fvp[t] * offer_energy_price[t] * delta_t for t in range(T)) - \
                gp.quicksum(P_Grid[t] * Grid_energy_price[t] * delta_t for t in range(T)) + \
                z_EV
    m.setObjective(objective, GRB.MAXIMIZE)

    # Optimize model
    m.optimize()

    optimal_values = {
        "P_total_demand": [P_fvp[t].X for t in range(T)],
        "P_total_dispatch": [P_Feed_in[t].X for t in range(T)],
        "Total_Objective_Value": m.ObjVal
    }

    return optimal_values
