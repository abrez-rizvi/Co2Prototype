import streamlit as st
import pandas as pd
import os

from data_manager import list_cities, load_city, load_custom_data, save_results
from simulation import run_simulation
from visualization import display_bar_chart, display_heatmap
from report_generator import generate_summary


def main():
    st.title('Digital Twin for CO₂ Capture Prototype')

    # Sidebar controls
    st.sidebar.header('Data Selection')
    
    # City selector
    cities = list_cities()
    if not cities:
        st.sidebar.warning('No preset city data found in data/ folder.')
        cities = []
    
    selected_city = st.sidebar.selectbox('Select preset city', [''] + cities, index=0)
    
    # File uploader for custom JSON
    uploaded_file = st.sidebar.file_uploader('Or upload custom JSON', type=['json'])
    
    # Load data button
    load_button = st.sidebar.button('Load Data')
    
    # Initialize session state for data
    if 'city_data' not in st.session_state:
        st.session_state.city_data = None
        st.session_state.city_name = None
    
    # Load data on button click
    if load_button:
        if uploaded_file is not None:
            try:
                st.session_state.city_data = load_custom_data(uploaded_file)
                st.session_state.city_name = st.session_state.city_data.get('city', 'Custom')
                st.sidebar.success(f'Loaded custom data: {st.session_state.city_name}')
            except Exception as e:
                st.sidebar.error(f'Failed to load custom file: {e}')
                st.session_state.city_data = None
        elif selected_city:
            try:
                st.session_state.city_data = load_city(selected_city)
                st.session_state.city_name = selected_city
                st.sidebar.success(f'Loaded city: {selected_city}')
            except Exception as e:
                st.sidebar.error(f'Failed to load city: {e}')
                st.session_state.city_data = None
        else:
            st.sidebar.warning('Please select a city or upload a file.')
    
    # Main view
    if st.session_state.city_data is None:
        st.info('👈 Please select a city or upload a custom JSON file from the sidebar, then click "Load Data".')
        st.markdown("""
        **Expected JSON format:**
        ```json
        {
          "city": "YourCity",
          "sectors": {
            "transport": 1200,
            "energy": 2200,
            "industry": 1500,
            "infrastructure": 800
          }
        }
        ```
        """)
        return
    
    city_data = st.session_state.city_data
    city_name = st.session_state.city_name
    baselines = city_data.get('sectors', {})
    
    if not baselines:
        st.error('No sector data found in loaded city data.')
        return
    
    # Display initial sector CO₂ chart
    st.header(f'Initial CO₂ Emissions — {city_name}')
    st.markdown('Baseline emissions by sector:')
    
    # Create a simple bar chart for baseline
    baseline_df = pd.DataFrame([
        {'Sector': s.capitalize(), 'CO₂ (units)': float(v)}
        for s, v in baselines.items()
    ])
    st.bar_chart(baseline_df.set_index('Sector'))
    
    # Sliders for each sector to apply % change
    st.header('Intervention Controls')
    st.markdown('Adjust sliders to simulate percentage changes in each sector (-50% to +50%):')
    
    interventions = {}
    cols = st.columns(2)
    sector_list = list(baselines.keys())
    for i, sector in enumerate(sector_list):
        col = cols[i % 2]
        with col:
            val = st.slider(
                f'{sector.capitalize()}',
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                help=f'Adjust CO₂ emissions for {sector} sector'
            )
            interventions[sector] = val / 100.0  # convert to decimal
    
    # Run Simulation button
    run_button = st.button('🚀 Run Simulation', type='primary', use_container_width=True)
    
    if run_button:
        # Run the simulation
        updated = run_simulation(baselines, interventions)
        
        # Build DataFrame for display
        df = pd.DataFrame([
            {'sector': s, 'baseline': float(baselines.get(s, 0.0)), 'simulated': float(updated.get(s, 0.0))}
            for s in baselines.keys()
        ]).set_index('sector')
        df['delta'] = df['baseline'] - df['simulated']
        df['pct_change'] = df['delta'] / df['baseline'].replace({0: 1}) * 100.0
        
        st.header('Simulation Results')
        
        # Updated bar chart
        st.subheader('Before vs After Comparison')
        display_bar_chart(baselines, updated, title='CO₂ Emissions: Baseline vs Simulated')
        
        # Heatmap
        st.subheader('Sector Intensity Heatmap')
        display_heatmap(updated, title='Simulated CO₂ Intensity by Sector')
        
        # Text summary
        st.subheader('Summary Report')
        text_summary = generate_summary(baselines, updated)
        st.info(text_summary)
        
        # Detailed table
        with st.expander('📊 View Detailed Data Table'):
            st.dataframe(df.style.format({
                'baseline': '{:.1f}',
                'simulated': '{:.1f}',
                'delta': '{:.1f}',
                'pct_change': '{:.1f}%'
            }))
        
        # Save Results button
        st.subheader('Export Results')
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('💾 Save Results (JSON + CSV)', use_container_width=True):
                try:
                    json_path, csv_path = save_results(city_name, df)
                    st.success(f'✅ Results saved!\n\n- JSON: `{os.path.basename(json_path)}`\n- CSV: `{os.path.basename(csv_path)}`')
                except Exception as e:
                    st.error(f'Failed to save results: {e}')
        
        with col2:
            # Download button for CSV
            csv_data = df.to_csv()
            st.download_button(
                label='⬇️ Download CSV',
                data=csv_data,
                file_name=f'{city_name}_simulation_results.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    # Sidebar info
    st.sidebar.markdown('---')
    st.sidebar.markdown('### How to Use')
    st.sidebar.markdown("""
    1. Select a preset city or upload custom JSON
    2. Click **Load Data**
    3. Adjust intervention sliders
    4. Click **Run Simulation**
    5. Review results and export if needed
    """)


if __name__ == '__main__':
    main()
