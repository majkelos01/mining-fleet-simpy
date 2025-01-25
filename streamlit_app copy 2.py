import streamlit as st
import simpy
import random
import statistics
import pandas as pd
import plotly.express as px
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MineSite:
    def __init__(self, env, num_trucks, excavator_loading_time,
                 segments, truck_capacity=180):
        self.env = env
        self.excavator = simpy.Resource(env, capacity=1)
        self.ramp = simpy.Resource(env, capacity=1)
        self.num_trucks = num_trucks
        self.loading_time = excavator_loading_time
        self.truck_capacity = truck_capacity
        self.segments = segments
        self.speeds = {
            'flat': 40,
            'flat_single_lane': 40,
            'ramp_up': 12,
            'ramp_down': 30,
            'ramp_single_lane_up': 12,
            'ramp_single_lane_down': 30
        }
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
            
            # Process loaded segments (to dump site) - direction 'up'
            direction = 'up'
            for segment in self.segments:
                start_time = self.env.now
                segment_type = segment['type']
                is_single_lane = '_single_lane' in segment_type
                base_type = segment_type.replace('_single_lane', '')
                
                # Determine speed based on segment type and direction
                try:
                    if 'ramp' in base_type:
                        speed_key = f"{base_type}_{direction}"
                        if is_single_lane:
                            speed_key = f"{base_type}_single_lane_{direction}"
                    else:
                        speed_key = base_type
                        if is_single_lane:
                            speed_key = f"{base_type}_single_lane"
                    
                    logger.debug(f"Processing segment: type={segment_type}, base={base_type}, direction={direction}, single_lane={is_single_lane}")
                    logger.debug(f"Generated speed key: {speed_key}")
                    
                    speed = self.speeds[speed_key]
                    logger.debug(f"Using speed: {speed} km/h for segment")
                except KeyError as e:
                    logger.error(f"Speed key error: {e}")
                    logger.error(f"Available speed keys: {list(self.speeds.keys())}")
                    raise
                travel_time = self.calculate_travel_time(segment['distance'], speed)
                yield self.env.timeout(random.normalvariate(travel_time, travel_time * 0.1))
                
                # Initialize segment time list if it doesn't exist
                segment_key = f"{base_type}_up" if 'up' in direction else f"{base_type}_down"
                if segment_key not in self.segment_times:
                    self.segment_times[segment_key] = []
                self.segment_times[segment_key].append(self.env.now - start_time)
                
                # Handle queues for single lane segments
                if is_single_lane and 'ramp' in base_type:
                    queue_start = self.env.now
                    with self.ramp.request() as ramp_access:
                        yield ramp_access
                        self.queue_times_ramp.append(self.env.now - queue_start)
            
            # Dumping process
            start_time = self.env.now
            yield self.env.timeout(2)
            self.segment_times['dumping'].append(self.env.now - start_time)
            
            # Process empty segments (return trip) - direction 'down'
            direction = 'down'
            for segment in reversed(self.segments):
                start_time = self.env.now
                segment_type = segment['type']
                is_single_lane = '_single_lane' in segment_type
                base_type = segment_type.replace('_single_lane', '')
                
                try:
                    if 'ramp' in base_type:
                        speed_key = f"{base_type}_down"
                        if is_single_lane:
                            speed_key = f"{base_type}_single_lane_down"
                    else:
                        speed_key = base_type
                        if is_single_lane:
                            speed_key = f"{base_type}_single_lane"
                    
                    logger.debug(f"Processing return segment: type={segment_type}, base={base_type}, direction=down, single_lane={is_single_lane}")
                    logger.debug(f"Generated speed key: {speed_key}")
                    
                    speed = self.speeds[speed_key]
                    logger.debug(f"Using speed: {speed} km/h for return segment")
                    
                    travel_time = self.calculate_travel_time(segment['distance'], speed)
                    yield self.env.timeout(random.normalvariate(travel_time, travel_time * 0.1))
                    
                    segment_key = f"{base_type}_down"
                    if segment_key not in self.segment_times:
                        self.segment_times[segment_key] = []
                    self.segment_times[segment_key].append(self.env.now - start_time)
                except KeyError as e:
                    logger.error(f"Return speed key error: {e}")
                    logger.error(f"Available speed keys: {list(self.speeds.keys())}")
                    raise
                
                if segment['type'] == 'ramp':
                    queue_start = self.env.now
                    with self.ramp.request() as ramp_access:
                        yield ramp_access
                        self.queue_times_ramp.append(self.env.now - queue_start)
            
            cycle_time = self.env.now - cycle_start
            self.cycle_times.append(cycle_time)

def run_simulation(num_trucks, segments, excavator_loading_time, truck_capacity, sim_time=720):
    env = simpy.Environment()
    mine = MineSite(
        env,
        num_trucks=num_trucks,
        excavator_loading_time=excavator_loading_time,
        segments=segments,
        truck_capacity=truck_capacity
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
        'segment_times': {k: statistics.mean(v) for k, v in mine.segment_times.items()},
        'truck_capacity': mine.truck_capacity
    }

def perform_sensitivity_analysis(min_trucks, max_trucks, segments, excavator_loading_time, truck_capacity):
    if not segments:
        raise ValueError("Please configure at least one haul road segment")
    
    # Check for single lane configuration
    ramp_count = sum(1 for s in segments if s['type'] == 'ramp')
    if ramp_count > 1:
        st.warning("Multiple ramp segments detected. Ensure your haul road has proper passing lanes.")
    
    results = []
    for trucks in range(min_trucks, max_trucks + 1):
        sim_result = run_simulation(
            trucks,
            segments,
            excavator_loading_time,
            truck_capacity
        )
        sim_result['num_trucks'] = trucks
        results.append(sim_result)
    return results

def main():
    st.set_page_config(page_title="Mining Truck Simulation", layout="wide")
    st.title("Mining Truck Fleet Optimization")
    
    with st.sidebar:
        st.header("Simulation Parameters")
        col1, col2 = st.columns(2)
        with col1:
            min_trucks = st.slider("Minimum Number of Trucks", 3, 20, 3)
            max_trucks = st.slider("Maximum Number of Trucks", 3, 20, 11)
            sim_time = st.number_input("Simulation Time (minutes)", 60, 1440, 720)
        with col2:
            excavator_loading_time = st.number_input("Excavator Loading Time (minutes)",
                min_value=1.0, max_value=10.0, value=4.0, step=0.1)
            truck_capacity = st.number_input("Truck Capacity (tonnes)",
                min_value=100, max_value=400, value=180, step=10)
        
        st.header("Haul Road Configuration")
        
        # Initialize segments in session state
        if 'segments' not in st.session_state:
            st.session_state.segments = []
            
        col1, col2 = st.columns(2)
        with col1:
            segment_type = st.selectbox("Segment Type",
                ["flat", "flat_single_lane", "ramp", "ramp_single_lane"],
                key="segment_type")
        with col2:
            segment_distance = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=1.0, step=0.1, key="segment_distance")
        
        add_col, clear_col = st.columns(2)
        with add_col:
            if st.button("Add Segment"):
                if len(st.session_state.segments) < 10:  # Limit to 10 segments
                    st.session_state.segments.append({
                        'type': segment_type,
                        'distance': segment_distance
                    })
                else:
                    st.warning("Maximum of 10 segments reached")
        
        if st.session_state.segments:
            st.subheader("Current Haul Road")
            for i, segment in enumerate(st.session_state.segments, 1):
                st.write(f"Segment {i}: {segment['distance']}km {segment['type']}")
            
            with clear_col:
                if st.button("Clear Segments"):
                    st.session_state.segments.clear()
    
    if st.sidebar.button("Run Simulation"):
        if not st.session_state.segments:
            st.error("Please configure at least one haul road segment")
        else:
            with st.spinner("Running simulation..."):
                try:
                    results = perform_sensitivity_analysis(
                        min_trucks,
                        max_trucks,
                        st.session_state.segments,
                        excavator_loading_time,
                        truck_capacity
                    )
                    df = pd.DataFrame(results)
                except ValueError as e:
                    st.error(f"Simulation error: {str(e)}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error during simulation: {str(e)}")
                    return
            
            st.header("Simulation Results")
            
            # Calculate truck productivity (tonnes/hour)
            df['truck_productivity'] = df['truck_capacity'] / (df['avg_cycle_time'] / 60)
            
            # Create three columns for graphs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Productivity Analysis")
                fig1 = px.line(df, x='num_trucks', y='productivity',
                             title='Total Productivity vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'productivity': 'Productivity (tonnes)'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.line(df, x='num_trucks', y='truck_productivity',
                             title='Truck Productivity vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'truck_productivity': 'Truck Productivity (t/h)'})
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.subheader("Cycle & Queue Times")
                fig3 = px.line(df, x='num_trucks', y='avg_cycle_time',
                             title='Cycle Time vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'avg_cycle_time': 'Cycle Time (minutes)'})
                st.plotly_chart(fig3, use_container_width=True)
                
                fig4 = px.line(df, x='num_trucks', y=['avg_queue_time_excavator', 'avg_queue_time_ramp'],
                             title='Queue Times vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'value': 'Queue Time (minutes)',
                                   'avg_queue_time_excavator': 'Excavator Queue',
                                   'avg_queue_time_ramp': 'Ramp Queue'})
                # Move legend to bottom and make it horizontal
                fig4.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ))
                st.plotly_chart(fig4, use_container_width=True)
            
            with col3:
                st.subheader("Excavator Analysis")
                fig5 = px.line(df, x='num_trucks', y='excavator_productivity',
                             title='Excavator Throughput vs Fleet Size',
                             labels={'num_trucks': 'Number of Trucks', 'excavator_productivity': 'Excavator Productivity (t/h)'})
                st.plotly_chart(fig5, use_container_width=True)
                
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