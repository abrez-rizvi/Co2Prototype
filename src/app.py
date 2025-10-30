import streamlit as st
import pandas as pd
import os
import json
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import random
from dotenv import load_dotenv

from data_manager import list_cities, load_city, load_custom_data, save_results
from simulation import run_simulation
from visualization import display_bar_chart
from report_generator import generate_summary
from geospatial_heatmap import display_before_after_heatmaps, display_difference_heatmap

# === Load Environment Variables ===
load_dotenv()

# === Configure Gemini ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# === Fix image folder path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def get_gemini_report(df, city_name):
    """Send DataFrame to Gemini and get a structured report."""
    data_json = df.reset_index().to_dict(orient="records")

    prompt = f"""
    You are an environmental analyst. Generate a detailed report on CO₂ emissions for {city_name}.
    Here is the emission data per sector: {json.dumps(data_json, indent=2)}

    Structure your report in this format:
    1. Overview
    2. Sector Analysis
    3. Recommendations
    4. Conclusion

    Keep the language analytical but simple.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"


def pick_random_image():
    """Pick a random existing image from images folder."""
    images = [
        os.path.join(IMAGES_DIR, "baseline_chart.png"),
        os.path.join(IMAGES_DIR, "baseline_heatmap.png"),
        os.path.join(IMAGES_DIR, "difference_heatmap.png"),
        os.path.join(IMAGES_DIR, "simulated_heatmap.png")
    ]
    valid_images = [img for img in images if os.path.exists(img)]
    if not valid_images:
        return None
    return random.choice(valid_images)


def main():
    st.title('🌍 Digital Twin for CO₂ Capture Prototype')

    # Sidebar controls
    st.sidebar.header('Data Selection')
    cities = list_cities()
    if not cities:
        st.sidebar.warning('No preset city data found in data/ folder.')
        cities = []

    selected_city = st.sidebar.selectbox('Select preset city', [''] + cities)
    uploaded_file = st.sidebar.file_uploader('Or upload custom JSON', type=['json'])
    load_button = st.sidebar.button('Load Data')

    # Session state
    if 'city_data' not in st.session_state:
        st.session_state.city_data = None
        st.session_state.city_name = None

    # Load data
    if load_button:
        if uploaded_file is not None:
            try:
                st.session_state.city_data = load_custom_data(uploaded_file)
                st.session_state.city_name = st.session_state.city_data.get('city', 'Custom')
                st.sidebar.success(f"Loaded custom data: {st.session_state.city_name}")
            except Exception as e:
                st.sidebar.error(f"Failed to load custom file: {e}")
        elif selected_city:
            try:
                st.session_state.city_data = load_city(selected_city)
                st.session_state.city_name = selected_city
                st.sidebar.success(f"Loaded city: {selected_city}")
            except Exception as e:
                st.sidebar.error(f"Failed to load city: {e}")
        else:
            st.sidebar.warning('Please select a city or upload a file.')

    if st.session_state.city_data is None:
        st.info("👈 Load a city dataset to begin.")
        st.markdown("""
        **Expected JSON:**
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
        st.error('No sector data found.')
        return

    # --- Baseline chart ---
    st.header(f'Initial CO₂ Emissions — {city_name}')
    baseline_df = pd.DataFrame([
        {'Sector': s.capitalize(), 'CO₂ (units)': float(v)} for s, v in baselines.items()
    ])

    # Save chart as image
    plt.figure(figsize=(6, 4))
    sns.barplot(data=baseline_df, x="Sector", y="CO₂ (units)", palette="Blues_d")
    plt.title(f"{city_name} - Baseline CO₂ Emissions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "baseline_chart.png"))
    plt.close()

    # Show chart interactively
    st.bar_chart(baseline_df.set_index('Sector'))

    # --- Intervention sliders ---
    st.header('Intervention Controls')
    st.markdown('Adjust % changes (-50% to +50%) per sector:')
    interventions = {}
    cols = st.columns(2)
    for i, sector in enumerate(baselines.keys()):
        col = cols[i % 2]
        with col:
            val = st.slider(f'{sector.capitalize()}', -50, 50, 0, 5)
            interventions[sector] = val / 100.0

    run_button = st.button('🚀 Run Simulation', type='primary', width='stretch')

    if run_button:
        updated = run_simulation(baselines, interventions)

        df = pd.DataFrame([
            {'sector': s, 'baseline': float(baselines[s]), 'simulated': float(updated[s])}
            for s in baselines
        ]).set_index('sector')
        df['delta'] = df['baseline'] - df['simulated']
        df['pct_change'] = df['delta'] / df['baseline'].replace({0: 1}) * 100.0

        st.header('Simulation Results')
        st.subheader('Before vs After Comparison')
        display_bar_chart(baselines, updated, title='CO₂ Emissions: Baseline vs Simulated')

        st.subheader('Geospatial Emission Distribution')
        display_before_after_heatmaps(baselines, updated, grid_size=(30, 30))

        st.subheader('Emission Change Map')
        display_difference_heatmap(baselines, updated, grid_size=(30, 30))

        with st.expander('📊 View Detailed Data Table'):
            st.dataframe(df.style.format({
                'baseline': '{:.1f}',
                'simulated': '{:.1f}',
                'delta': '{:.1f}',
                'pct_change': '{:.1f}%'
            }))

        # --- Gemini AI Report ---
        st.header('🧠 Gemini AI Summary Report')
        with st.spinner('Generating AI report...'):
            report_text = get_gemini_report(df, city_name)

        # Split into sections
        sections = re.split(r"(?i)(?:\n|^)\s*\d+\.?\s*(overview|sector analysis|recommendations|conclusion)", report_text)
        sections = [s.strip() for s in sections if s.strip()]

        # Section 1 — Overview
        st.subheader("1️⃣ Overview")
        st.markdown(sections[0] if len(sections) > 0 else "Overview section missing.")
        img = pick_random_image()
        print(img)
        if img:
            st.image(img, caption="Overview Visualization", width='stretch')

        # Section 2 — Sector Analysis
        if len(sections) > 1:
            st.subheader("2️⃣ Sector Analysis")
            st.markdown(sections[1])
            img = pick_random_image()
            if img:
                st.image(img, caption="Sector Analysis Visualization", width='stretch')

        # Section 3 — Recommendations
        if len(sections) > 2:
            st.subheader("3️⃣ Recommendations")
            st.markdown(sections[2])
            img = pick_random_image()
            if img:
                st.image(img, caption="Recommendations Visualization", width='stretch')

        # Section 4 — Conclusion
        if len(sections) > 3:
            st.subheader("4️⃣ Conclusion")
            st.markdown(sections[3])
            img = pick_random_image()
            if img:
                st.image(img, caption="Conclusion Visualization", width='stretch')

        # --- Save Results ---
        st.subheader('Export Results')
        if st.button('💾 Save Results'):
            json_path, csv_path = save_results(city_name, df)
            st.success(f"Saved to:\n- {json_path}\n- {csv_path}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Steps:**
    1. Load data  
    2. Adjust sliders  
    3. Run Simulation  
    4. Review Results  
    5. Generate AI Report  
    6. Export if needed
    """)


if __name__ == '__main__':
    main()
