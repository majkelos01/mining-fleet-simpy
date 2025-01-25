import simpy
import random
import statistics
import matplotlib.pyplot as plt
import pandas as pd

class MineSite:
    def __init__(self, env, num_trucks, excavator_loading_time, 
                 ramp_up_distance, ramp_down_distance, 
                 flat_up_distance, flat_down_distance,
                 truck_capacity=180):  # Default capacity 180t
        self.env = env
        # Resources
        self.excavator = simpy.Resource(env, capacity=1)  # Single excavator
        self.ramp = simpy.Resource(env, capacity=1)       # Single lane ramp
        
        # Parameters
        self.num_trucks = num_trucks
        self.loading_time = excavator_loading_time
        self.truck_capacity = truck_capacity
        
        # Distances (km)
        self.ramp_up_distance = ramp_up_distance
        self.ramp_down_distance = ramp_down_distance
        self.flat_up_distance = flat_up_distance
        self.flat_down_distance = flat_down_distance
        
        # Speeds (km/h)
        self.flat_speed = 40  # km/h
        self.ramp_up_speed = 12  # km/h
        self.ramp_down_speed = 30  # km/h
        
        # Statistics
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
        # New productivity metrics
        self.total_tonnes_loaded = 0
        self.excavator_loading_events = 0

    def calculate_travel_time(self, distance, speed):
        """Calculate travel time in minutes from distance (km) and speed (km/h)"""
        return (distance / speed) * 60  # Convert hours to minutes

    def truck_cycle(self, truck_id):
        while True:
            cycle_start = self.env.now
            
            # 1. Loading at excavator
            queue_start = self.env.now
            with self.excavator.request() as request:
                yield request
                queue_time = self.env.now - queue_start
                self.queue_times_excavator.append(queue_time)
                
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(self.loading_time, self.loading_time * 0.1))
                self.segment_times['loading'].append(self.env.now - start_time)
                # Track excavator productivity
                self.total_tonnes_loaded += self.truck_capacity
                self.excavator_loading_events += 1
            
            # 2. Hauling up (flat section first, then ramp)
            # Flat section (unlimited capacity)
            start_time = self.env.now
            flat_time = self.calculate_travel_time(self.flat_up_distance, self.flat_speed)
            yield self.env.timeout(random.normalvariate(flat_time, flat_time * 0.1))
            self.segment_times['flat_up'].append(self.env.now - start_time)
            
            # Ramp section (single lane)
            queue_start = self.env.now
            with self.ramp.request() as ramp_access:
                yield ramp_access
                self.queue_times_ramp.append(self.env.now - queue_start)
                
                # Calculate ramp haul time
                ramp_time = self.calculate_travel_time(self.ramp_up_distance, self.ramp_up_speed)
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(ramp_time, ramp_time * 0.1))
                self.segment_times['ramp_up'].append(self.env.now - start_time)
            
            # 3. Dumping
            start_time = self.env.now
            yield self.env.timeout(2)  # Fixed dumping time
            self.segment_times['dumping'].append(self.env.now - start_time)
            
            # 4. Return journey (ramp first, then flat)
            # Ramp section
            with self.ramp.request() as ramp_access:
                yield ramp_access
                # Calculate ramp return time
                ramp_time = self.calculate_travel_time(self.ramp_down_distance, self.ramp_down_speed)
                start_time = self.env.now
                yield self.env.timeout(random.normalvariate(ramp_time, ramp_time * 0.1))
                self.segment_times['ramp_down'].append(self.env.now - start_time)
            
            # Flat section (unlimited capacity)
            start_time = self.env.now
            flat_time = self.calculate_travel_time(self.flat_down_distance, self.flat_speed)
            yield self.env.timeout(random.normalvariate(flat_time, flat_time * 0.1))
            self.segment_times['flat_down'].append(self.env.now - start_time)
            
            # Record cycle completion
            cycle_time = self.env.now - cycle_start
            self.cycle_times.append(cycle_time)

def run_simulation(num_trucks, sim_time=720):  # 480 minutes = 8 hours
    env = simpy.Environment()
    
    # Create mine site with parameters
    mine = MineSite(
        env,
        num_trucks=num_trucks,
        excavator_loading_time=4,      # 3 minutes loading
        ramp_up_distance=0.6,            # 2 km up ramp
        ramp_down_distance=0.6,          # 2 km down ramp
        flat_up_distance=5,            # 5 km flat haul
        flat_down_distance=5,          # 5 km flat return
        truck_capacity=180             # 180t capacity
    )
    
    # Start truck processes
    for i in range(num_trucks):
        env.process(mine.truck_cycle(f'Truck_{i}'))
    
    # Run simulation
    env.run(until=sim_time)
    
    # Calculate excavator productivity (tonnes/hour)
    excavator_productivity = (mine.total_tonnes_loaded / sim_time) * 60
    
    return {
        'avg_cycle_time': statistics.mean(mine.cycle_times),
        'avg_queue_time_excavator': statistics.mean(mine.queue_times_excavator),
        'avg_queue_time_ramp': statistics.mean(mine.queue_times_ramp),
        'total_cycles': len(mine.cycle_times),
        'productivity': mine.total_tonnes_loaded,  # Total tonnes moved
        'excavator_productivity': excavator_productivity,
        'segment_times': {k: statistics.mean(v) for k, v in mine.segment_times.items()}
    }

def perform_sensitivity_analysis():
    results = []
    for trucks in range(3, 12):
        sim_result = run_simulation(trucks)
        sim_result['num_trucks'] = trucks
        results.append(sim_result)
    return results

def plot_results(results):
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Create multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Productivity plot
    ax1.plot(df['num_trucks'], df['productivity'], 'b-o')
    ax1.set_xlabel('Number of Trucks')
    ax1.set_ylabel('Productivity (tonnes)')
    ax1.set_title('Total Productivity vs Fleet Size')
    ax1.grid(True)
    
    # Excavator productivity plot
    ax2.plot(df['num_trucks'], df['excavator_productivity'], 'g-o')
    ax2.set_xlabel('Number of Trucks')
    ax2.set_ylabel('Excavator Productivity (t/h)')
    ax2.set_title('Excavator Throughput vs Fleet Size')
    ax2.grid(True)
    
    # Queue times plot
    ax3.plot(df['num_trucks'], df['avg_queue_time_excavator'], 'g-o', label='Excavator')
    ax3.plot(df['num_trucks'], df['avg_queue_time_ramp'], 'r-o', label='Ramp')
    ax3.set_xlabel('Number of Trucks')
    ax3.set_ylabel('Average Queue Time (minutes)')
    ax3.set_title('Queue Times vs Fleet Size')
    ax3.legend()
    ax3.grid(True)
    
    # Segment times for selected fleet size
    mid_fleet = df.iloc[len(df)//2]
    segment_times = mid_fleet['segment_times']
    segments = list(segment_times.keys())
    times = list(segment_times.values())
    ax4.bar(segments, times)
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Time (minutes)')
    ax4.set_title(f'Segment Times (Fleet Size: {mid_fleet["num_trucks"]})')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = perform_sensitivity_analysis()
    plot_results(results)
    
    # Print detailed results for optimal fleet size
    df_results = pd.DataFrame(results)
    optimal_fleet = df_results.loc[df_results['productivity'].idxmax()]
    
    print("\nOptimal Fleet Configuration:")
    print(f"Number of trucks: {optimal_fleet['num_trucks']}")
    print(f"Total productivity: {optimal_fleet['productivity']:.0f} tonnes")
    print(f"Excavator productivity: {optimal_fleet['excavator_productivity']:.1f} t/h")
    print(f"Average cycle time: {optimal_fleet['avg_cycle_time']:.2f} minutes")
    print("\nQueue times:")
    print(f"Excavator: {optimal_fleet['avg_queue_time_excavator']:.2f} minutes")
    print(f"Ramp: {optimal_fleet['avg_queue_time_ramp']:.2f} minutes")