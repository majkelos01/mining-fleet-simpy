import streamlit as st
import simpy
import random
import statistics
import pandas as pd
import plotly.express as px

class MineSite:
    def __init__(self, env, num_trucks, excavator_loading_time, 
                 ramp_up_distance, ramp_down_distance, 
                 flat_up_distance, flat_down_distance,
                 truck_capacity=180):
        self.env = env
        self.excavator = simpy.Resource(env, capacity=1)
        self.ramp = simpy.Resource(env, capacity=1)
        self.num_trucks = num_trucks
        self.loading_time = excavator_loading_time
        self.truck_capacity = truck_capacity
        self.ramp_up_distance = ramp_up_distance
        self.ramp_down_distance = ramp_down_distance
        self.flat_up_distance = flat_up_distance
        self.flat_down_distance = flat_down_distance
        self.flat_speed = 40
        self.ramp_up_speed = 12
        self.ramp_down_speed = 30
        self.cycle_times = []
        self.queue_times_excavator = []
        self.queue_times_ramp = []
        self.segment_times = {
            'loading': [],
            'ramp_up': [],
            'flat_up': [],
            'dumping': [],
            'ramp_down': [],
            'flat_down': []
        }
        self.total_tonnes_loaded = 0
        self.excavator_loading_events = 0

    def calculate_travel_time(self, distance, speed):
        return (distance / speed) * 60

    def truck_cycle(self, truck_id):
        while True:
            cycle_start = self.env.now
            queue_start = self.env.now
            with self.excavator.request() as request:
                yield request
                queue_time = self.env.now - queue_start
                self.queue_times_excavator.append(queue_time)
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(self.loading_time, self.loading_time * 0.1))
                self.segment_times['loading'].append(self.env.now - start_time)
                self.total_tonnes_loaded += self.truck_capacity
                self.excavator_loading_events += 1
            
            start_time = self.env.now
            flat_time = self.calculate_travel_time(self.flat_up_distance, self.flat_speed)
            yield self.env.timeout(random.normalvariate(flat_time, flat_time * 0.1))
            self.segment_times['flat_up'].append(self.env.now - start_time)
            
            queue_start = self.env.now
            with self.ramp.request() as ramp_access:
                yield ramp_access
                self.queue_times_ramp.append(self.env.now - queue_start)
                ramp_time = self.calculate_travel_time(self.ramp_up_distance, self.ramp_up_speed)
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(ramp_time, ramp_time * 0.1))
                self.segment_times['ramp_up'].append(self.env.now - start_time)
            
            start_time = self.env.now
            yield self.env.timeout(2)
            self.segment_times['dumping'].append(self.env.now - start_time)
            
            with self.ramp.request() as ramp_access:
                yield ramp_access
                ramp_time = self.calculate_travel_time(self.ramp_down_distance, self.ramp_down_speed)
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(ramp_time, ramp_time * 0.1))
                self.segment_times['ramp_down'].append(self.env.now - start_time)
            
            start_time = self.env.now
            flat_time = self.calculate_travel_time(self.flat_down_distance, self.flat_speed)
            yield self.env.timeout(random.normalvariate(flat_time, flat_time * 0.1))
            self.segment_times['flat_down'].append(self.env.now - start_time)
            
            cycle_time = self.env.now - cycle_start
            self.cycle_times.append(cycle_time)

def run_simulation(num_trucks, sim_time=720):
    env = simpy.Environment()
    mine = MineSite(
        env,
        num_trucks=num_trucks,
        excavator_loading_time=4,
        ramp_up_distance=0.6,
        ramp_down_distance=0.6,
        flat_up_distance=5,
        flat_down_distance=5,
        truck_capacity=180
    )
    
    for i in range(num_trucks):
        env.process(mine.truck_cycle(f'Truck_{i}'))
    
    env.run(until=sim_time)
    excavator_productivity = (mine.total_tonnes_loaded / sim_time) * 60
    
    return {
        'avg_cycle_time': statistics.mean(mine.cycle_times),
        'avg_queue_time_excavator': statistics.mean(mine.queue_times_excavator),
        'avg_queue_time_ramp': statistics.mean(mine.queue_times_ramp),
        'total_cycles': len(mine.cycle_times),
        'productivity': mine.total_tonnes_loaded,
        'excavator_productivity': excavator_productivity,
        'segment_times': {k: statistics.mean(v) for k, v in mine.segment_times.items()}
    }

def perform_sensitivity_analysis(min_trucks, max_trucks):
    results = []
    for trucks in range(min_trucks, max_trucks + 1):
        sim_result = run_simulation(trucks)
        sim_result['num_trucks'] = trucks
        results.append(sim_result)
    return results

def main():
    st.set_page_config(page_title="Mining Truck Simulation", layout="wide")
    st.title("Mining Truck Fleet Optimization")
    
    with st.sidebar:
        st.header("Simulation Parameters")
        min_trucks = st.slider("Minimum Number of Trucks", 3, 20, 3)
        max_trucks = st.slider("Maximum Number of Trucks", 3, 20, 11)
        sim_time = st.number_input("Simulation Time (minutes)", 60, 1440, 720)
    
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            results = perform_sensitivity_analysis(min_trucks, max_trucks)
            df = pd.DataFrame(results)
            
            st.header("Simulation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Productivity Analysis")
                fig1 = px.line(df, x='num_trucks', y='productivity',
                             title='Total Productivity vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'productivity': 'Productivity (tonnes)'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.line(df, x='num_trucks', y='excavator_productivity',
                             title='Excavator Throughput vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'excavator_productivity': 'Excavator Productivity (t/h)'})
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.subheader("Queue Time Analysis")
                fig3 = px.line(df, x='num_trucks', y=['avg_queue_time_excavator', 'avg_queue_time_ramp'],
                             title='Queue Times vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'value': 'Queue Time (minutes)'})
                st.plotly_chart(fig3, use_container_width=True)
                
                st.subheader("Optimal Fleet Configuration")
                optimal_fleet = df.loc[df['productivity'].idxmax()]
                st.metric("Optimal Number of Trucks", optimal_fleet['num_trucks'])
                st.metric("Total Productivity", f"{optimal_fleet['productivity']:.0f} tonnes")
                st.metric("Excavator Productivity", f"{optimal_fleet['excavator_productivity']:.1f} t/h")
                st.metric("Average Cycle Time", f"{optimal_fleet['avg_cycle_time']:.2f} minutes")
                
                st.write("Queue Times:")
                st.write(f"- Excavator: {optimal_fleet['avg_queue_time_excavator']:.2f} minutes")
                st.write(f"- Ramp: {optimal_fleet['avg_queue_time_ramp']:.2f} minutes")

if __name__ == "__main__":
    main()