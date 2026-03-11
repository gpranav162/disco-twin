import simpy
import random
import pandas as pd
import os
from typing import Dict

class SupplyChainTwin:
    def __init__(self, env: simpy.Environment):
        self.env = env
        
        self.supplier = simpy.Resource(env, capacity=1)
        self.manufacturer = simpy.Resource(env, capacity=1)
        self.distributor = simpy.Resource(env, capacity=1)
        
        self.mu_1 = 18.0  
        self.mu_2 = 19.0  
        self.mu_3 = 20.0  
        
        # Operational flags for capacity loss disruptions
        self.supplier_up = True
        self.manufacturer_up = True
        self.distributor_up = True
        
        self.log = []
        
    def process_order(self, order_id: int):
        arrival_time = self.env.now
        
        # 1. Supplier Stage
        with self.supplier.request() as req:
            yield req
            # Halt processing if disrupted
            while not self.supplier_up:
                yield self.env.timeout(1)
            proc_time_1 = random.expovariate(self.mu_1)
            yield self.env.timeout(proc_time_1)
            
        # 2. Manufacturer Stage
        with self.manufacturer.request() as req:
            yield req
            while not self.manufacturer_up:
                yield self.env.timeout(1)
            proc_time_2 = random.expovariate(self.mu_2)
            yield self.env.timeout(proc_time_2)
            
        # 3. Distributor Stage
        with self.distributor.request() as req:
            yield req
            while not self.distributor_up:
                yield self.env.timeout(1)
            proc_time_3 = random.expovariate(self.mu_3)
            yield self.env.timeout(proc_time_3)
            
        finish_time = self.env.now
        
        # Logging core features
        self.log.append({
            'time': finish_time,
            'interarrival_time': arrival_time, # Will compute diffs later
            'supplier_proc_time': proc_time_1,
            'manufacturer_proc_time': proc_time_2,
            'distributor_proc_time': proc_time_3,
            'supplier_queue': len(self.supplier.queue),
            'manufacturer_queue': len(self.manufacturer.queue),
            'distributor_queue': len(self.distributor.queue),
            'lead_time': finish_time - arrival_time
        })

def order_generator(env: simpy.Environment, sc_twin: SupplyChainTwin, demand_params: Dict):
    order_id = 0
    while True:
        interarrival_time = random.expovariate(demand_params['rate'])
        yield env.timeout(interarrival_time)
        order_id += 1
        env.process(sc_twin.process_order(order_id))

def trigger_capacity_disruption(env: simpy.Environment, sc_twin: SupplyChainTwin, echelon: str):
    occurrence = random.uniform(300, 600)
    duration = random.uniform(30, 60)
    
    yield env.timeout(occurrence)
    setattr(sc_twin, f"{echelon}_up", False)
    
    yield env.timeout(duration)
    setattr(sc_twin, f"{echelon}_up", True)

def trigger_demand_surge(env: simpy.Environment, demand_params: Dict):
    occurrence = random.uniform(300, 600)
    duration = random.uniform(30, 60)
    
    yield env.timeout(occurrence)
    demand_params['rate'] = 30.0  # Surge rate
    
    yield env.timeout(duration)
    demand_params['rate'] = 15.0  # Normal rate

def run_simulation(scenario: int = 0, simulation_days: int = 1095, save_path: str = '../data/raw/s0_normal.csv'):
    env = simpy.Environment()
    sc_twin = SupplyChainTwin(env)
    
    demand_params = {'rate': 15.0} # Normal arrival rate
    env.process(order_generator(env, sc_twin, demand_params))
    
    # Inject Disruption Scenarios
    if scenario == 1:
        env.process(trigger_capacity_disruption(env, sc_twin, 'supplier'))
    elif scenario == 2:
        env.process(trigger_capacity_disruption(env, sc_twin, 'manufacturer'))
    elif scenario == 3:
        env.process(trigger_capacity_disruption(env, sc_twin, 'distributor'))
    elif scenario == 4:
        env.process(trigger_demand_surge(env, demand_params))
        
    env.run(until=simulation_days)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(sc_twin.log)
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Compute interarrival time properly
    df['interarrival_time'] = df['interarrival_time'].diff().fillna(0)
    
    df.to_csv(save_path, index=False)
    print(f"Scenario S{scenario} complete. Saved to {save_path}")

if __name__ == '__main__':
    # Generate data for all 5 scenarios
    for i in range(5):
        run_simulation(scenario=i, save_path=f'../data/raw/s{i}_data.csv')