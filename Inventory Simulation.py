import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Helper function to sample from a probability distribution
def sample_from(values, probs):
    r = random.random()
    cum = 0.0
    for v, p in zip(values, probs):
        cum += p
        if r <= cum:
            return v
    return values[-1]


# Simulation function
def simulate(days, reorder_interval, order_qtyA, order_qtyB, seed):
    random.seed(seed)
    showroom_transfer_qty = 5
    inventory_capacity = 30

    # Demand and lead time distributions
    demandA_values = [1, 2, 3, 4]
    demandA_probabilities = [0.15, 0.25, 0.25, 0.35]
    demandB_values = [1, 2, 3]
    demandB_probabilities = [0.30, 0.30, 0.40]
    leadTime_values = [1, 2, 3]
    leadTime_probabilities = [0.40, 0.35, 0.25]

    # Theoretical means
    theo_mean_DA = sum(v * p for v, p in zip(demandA_values, demandA_probabilities))
    theo_mean_DB = sum(v * p for v, p in zip(demandB_values, demandB_probabilities))
    theo_mean_lt = sum(v * p for v, p in zip(leadTime_values, leadTime_probabilities))

    # Initial inventory and showroom levels
    end_showroomA, end_showroomB = 5, 5
    end_inventoryA, end_inventoryB = 15, 15

    outstanding_orders = []
    pending_requestA, pending_requestB = False, False

    records = []
    lead_samples = []
    shortage_days_A = 0
    shortage_days_B = 0
    shortage_days_in_Both = 0

    # Simulation loop (day by day)
    for day in range(1, days + 1):
        start_showroomA = end_showroomA
        start_showroomB = end_showroomB
        start_inventoryA = end_inventoryA
        start_inventoryB = end_inventoryB

        # Transfer from inventory to showroom if previous request
        if pending_requestA:
            can_transfer = min(showroom_transfer_qty, start_inventoryA)
            start_showroomA += can_transfer
            end_inventoryA = start_inventoryA - can_transfer
            pending_requestA = False
        if pending_requestB:
            can_transfer = min(showroom_transfer_qty, start_inventoryB)
            start_showroomB += can_transfer
            end_inventoryB = start_inventoryB - can_transfer
            pending_requestB = False

        # Generate random daily demands
        dA = sample_from(demandA_values, demandA_probabilities)
        dB = sample_from(demandB_values, demandB_probabilities)

        # Serve demands from showroom
        servedA = min(start_showroomA, dA)
        shortageA = dA - servedA
        end_showroomA = start_showroomA - servedA

        servedB = min(start_showroomB, dB)
        shortageB = dB - servedB
        end_showroomB = start_showroomB - servedB

        # Count shortage days
        if shortageA > 0: shortage_days_A += 1
        if shortageB > 0: shortage_days_B += 1
        if shortageA > 0 or shortageB > 0: shortage_days_in_Both += 1

        # Request inventory if showroom is empty
        if end_showroomA == 0: pending_requestA = True
        if end_showroomB == 0: pending_requestB = True

        # Receive orders that have arrived
        arrivals = [o for o in outstanding_orders if o['arrival_day'] == day]
        for o in arrivals:
            end_inventoryA += o['qtyA']
            end_inventoryB += o['qtyB']

        # Periodic order review
        if day % reorder_interval == 0:
            lt = sample_from(leadTime_values, leadTime_probabilities)
            lead_samples.append(lt)
            arrival_day = day + lt

            free_space = inventory_capacity - (end_inventoryA + end_inventoryB)
            orderA = min(order_qtyA, free_space)
            free_space -= orderA
            orderB = min(order_qtyB, free_space)

            outstanding_orders.append({'arrival_day': arrival_day, 'qtyA': orderA, 'qtyB': orderB, 'lead_time': lt})

        # Record day stats
        records.append({
            'day': day,
            'start_showroomA': start_showroomA,
            'start_showroomB': start_showroomB,
            'start_inventoryA': start_inventoryA,
            'start_inventoryB': start_inventoryB,
            'end_showroomA': end_showroomA,
            'end_showroomB': end_showroomB,
            'end_inventoryA': end_inventoryA,
            'end_inventoryB': end_inventoryB,
            'servedA': servedA,
            'servedB': servedB,
            'demandA': dA,
            'demandB': dB,
            'shortageA': shortageA,
            'shortageB': shortageB,
        })

    df = pd.DataFrame(records)

    results = {
        'df': df,
        'avg_showroom': (df['end_showroomA'] + df['end_showroomB']).mean(),
        'avg_inventory': (df['end_inventoryA'] + df['end_inventoryB']).mean(),
        'shortage_days_A': shortage_days_A,
        'shortage_days_B': shortage_days_B,
        'shortage_days_in_Both': shortage_days_in_Both,
        'avg_demandA_exp': df['demandA'].mean(),
        'avg_demandB_exp': df['demandB'].mean(),
        'avg_lead_exp': (sum(lead_samples) / len(lead_samples)) if lead_samples else 0.0,
        'theo_mean_A': theo_mean_DA,
        'theo_mean_B': theo_mean_DB,
        'theo_mean_lead': theo_mean_lt
    }
    return results


def get_stats(data):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + 0.95) / 2., n - 1)
    return mean, mean - h, mean + h


# --- 1. RUN SIMULATION FOR INITIAL SUMMARY AND SNAPSHOT ---
res = simulate(days=15, reorder_interval=4, order_qtyA=6, order_qtyB=6, seed=42)
df_15 = res['df']

print("=== Simulation Summary for N=4, mA=6, mB=6 ===")
summary = {
    'Average end showroom (units)': res['avg_showroom'],
    'Average end inventory (units)': res['avg_inventory'],
    'Shortage days (A)': res['shortage_days_A'],
    'Shortage days (B)': res['shortage_days_B'],
    'Shortage days (any product)': res['shortage_days_in_Both'],
    'Experimental mean demand A': res['avg_demandA_exp'],
    'Experimental mean demand B': res['avg_demandB_exp'],
    'Theoretical mean demand A': res['theo_mean_A'],
    'Theoretical mean demand B': res['theo_mean_B'],
    'Experimental mean lead time': res['avg_lead_exp'],
    'Theoretical mean lead time': res['theo_mean_lead']
}
print(pd.DataFrame(list(summary.items()), columns=['Measure', 'Value']).to_string(index=False))

print("\n=== First 15 days snapshot (end-of-day) ===")
cols = ['day', 'start_showroomA', 'start_showroomB', 'start_inventoryA', 'start_inventoryB',
        'end_showroomA', 'end_showroomB', 'end_inventoryA', 'end_inventoryB',
        'servedA', 'servedB', 'demandA', 'demandB', 'shortageA', 'shortageB']
print(df_15[cols].head(15).to_string(index=False))

# --- 2. OPTIMIZATION OF N (50 REPLICATIONS) ---
NUM_REPS = 50
SIM_DAYS = 500
BASE_SEED = 42

print("\n=== Optimization of N (Review Period) with 50 Replications each ===")
n_opt_results = []
for N in range(1, 8):
    rep_shortages = []
    for i in range(NUM_REPS):
        r = simulate(SIM_DAYS, N, 6, 6, BASE_SEED + i)
        rep_shortages.append(r['shortage_days_in_Both'])

    mean, lb, ub = get_stats(rep_shortages)
    n_opt_results.append({'N': N, 'Mean Shortage': mean, 'LB': lb, 'UB': ub})

df_n_opt = pd.DataFrame(n_opt_results)
print(df_n_opt.to_string(index=False))

best_N = int(df_n_opt.loc[df_n_opt['Mean Shortage'].idxmin(), 'N'])
print(f"\n>>> Best review period N to minimize shortages: N = {best_N}")

# --- 3. SENSITIVITY ANALYSIS FOR mA AND mB (50 REPLICATIONS) ---
print("\n=== Order Quantity Combinations Optimization (using Best N and 50 Replications) ===")
m_results = []
for mA in range(4, 8):
    for mB in range(4, 8):
        rep_shortages = []
        for i in range(NUM_REPS):
            r = simulate(SIM_DAYS, best_N, mA, mB, BASE_SEED + i)
            rep_shortages.append(r['shortage_days_in_Both'])

        mean, lb, ub = get_stats(rep_shortages)
        m_results.append({'mA': mA, 'mB': mB, 'Mean Shortage': mean, 'LB': lb, 'UB': ub})

df_m_opt = pd.DataFrame(m_results).sort_values("Mean Shortage")
print(df_m_opt.to_string(index=False))


df_15['total_showroom'] = df_15['end_showroomA'] + df_15['end_showroomB']
df_15['total_inventory'] = df_15['end_inventoryA'] + df_15['end_inventoryB']

# Plot 1: Showroom
plt.figure(figsize=(8, 4))
plt.plot(df_15['day'], df_15['total_showroom'], marker='o', color='blue', label='Showroom Units')
plt.title('Graph 1: Total Showroom Units (First 15 Days)')
plt.xlabel('Day')
plt.ylabel('Units')
plt.grid(True)
plt.savefig('showroom_15days.png')

# Plot 2: Inventory
plt.figure(figsize=(8, 4))
plt.plot(df_15['day'], df_15['total_inventory'], marker='s', color='green', label='Inventory Units')
plt.title('Graph 2: Total Inventory Units (First 15 Days)')
plt.xlabel('Day')
plt.ylabel('Units')
plt.grid(True)
plt.savefig('inventory_15days.png')

plt.figure(figsize=(8, 5))
plt.errorbar(df_n_opt['N'], df_n_opt['Mean Shortage'],
             yerr=[df_n_opt['Mean Shortage']-df_n_opt['LB'], df_n_opt['UB']-df_n_opt['Mean Shortage']],
             fmt='-o', capsize=5, color='red', ecolor='black')
plt.title('Graph 3: Optimization of N (Average Shortage Days with 95% CI)')
plt.xlabel('Review Period (N)')
plt.ylabel('Average Shortage Days')
plt.grid(True)
plt.savefig('n_optimization_ci.png')
