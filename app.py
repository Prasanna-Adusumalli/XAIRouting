import pandas as pd
import streamlit as st
import random
import os
import streamlit.components.v1 as st_components

from gavrptw.core import run_gavrptw

# Set a seed for consistency
random.seed(64)

# Streamlit app title
st.title("Genetic Algorithm for VRPTW - Custom Instance Runner")

# Sidebar inputs for the instance parameters
st.sidebar.title("Algorithm Parameters")

# Instance selection (you can customize this with a dropdown for multiple instances if needed)
instance_name = st.sidebar.text_input("Instance Name", value="C204")

# Input fields for VRPTW costs
unit_cost = st.sidebar.number_input("Unit Cost", value=8.0)
init_cost = st.sidebar.number_input("Initial Cost", value=100.0)
wait_cost = st.sidebar.number_input("Waiting Cost", value=1.0)
delay_cost = st.sidebar.number_input("Delay Cost", value=1.5)

# Input fields for algorithm configuration
ind_size = st.sidebar.number_input("Individual Size", min_value=1, value=100)
pop_size = st.sidebar.number_input("Population Size", min_value=1, value=400)
cx_pb = st.sidebar.slider("Crossover Probability", min_value=0.0, max_value=1.0, value=0.85)
mut_pb = st.sidebar.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=0.02)
n_gen = st.sidebar.number_input("Number of Generations", min_value=1, value=300)

# Checkbox to export results to CSV
export_csv = st.sidebar.checkbox("Export CSV", value=True)

# Button to run the algorithm
if st.sidebar.button("Run Algorithm"):
    st.write(f"Running the Genetic Algorithm for instance {instance_name}...")

    # Call the run_gavrptw function with user inputs
    lime_html_path, shap_html_path, shap_summary_plot_path, route_image_path=run_gavrptw(instance_name=instance_name, unit_cost=unit_cost, init_cost=init_cost, \
                wait_cost=wait_cost, delay_cost=delay_cost, ind_size=ind_size, pop_size=pop_size, \
                cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)

    st.write("Algorithm finished running!")

    print(route_image_path)

    # Display the final routes plot at the beginning
    if route_image_path and os.path.exists(route_image_path):
        st.image(route_image_path, caption="Final Optimized Routes", use_column_width=True)

    # Display the LIME explanation
    if os.path.exists(lime_html_path):
        with open(lime_html_path, 'r', encoding='utf-8') as f:
            lime_html_content = f.read()
        st.write("### LIME Explanation for Best Individual")
        st_components.html(lime_html_content, height=800)

    # Display the SHAP explanation
    if os.path.exists(shap_html_path):
        with open(shap_html_path, 'r', encoding='utf-8') as f:
            shap_html_content = f.read()
        st.write("### SHAP Explanation for Best Individual")
        st_components.html(shap_html_content, height=800)

    # Display the SHAP summary plot
    if os.path.exists(shap_summary_plot_path):
        st.write("### SHAP Summary Plot")
        st.image(shap_summary_plot_path)

    # Optionally, check if results are saved and display them (optional)
    results_dir = os.path.join("results")  # Assuming results are saved here
    if os.path.exists(results_dir):
        st.write("Results are saved in the 'results' directory.")
        if export_csv:
            csv_file = os.path.join(results_dir, f"{instance_name}_results.csv")
            if os.path.exists(csv_file):
                st.write(f"CSV file saved at {csv_file}")
                # Optionally, display the first few rows of the CSV
                st.write("Preview of CSV Results:")
                st.dataframe(pd.read_csv(csv_file).head())