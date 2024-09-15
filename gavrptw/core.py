# -*- coding: utf-8 -*-

'''gavrptw/core.py'''
import itertools
import os
import io
import random
from csv import DictWriter

import lime
import numpy as np
import shap
from deap import base, creator, tools
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules


def ind2route(individual, instance):
    '''gavrptw.core.ind2route(individual, instance)'''
    route = []
    vehicle_capacity = instance['vehicle_capacity']
    depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    vehicle_load = 0
    elapsed_time = 0
    last_customer_id = 0
    for customer_id in individual:
        # Update vehicle load
        demand = instance[f'customer_{customer_id}']['demand']
        updated_vehicle_load = vehicle_load + demand
        # Update elapsed time
        service_time = instance[f'customer_{customer_id}']['service_time']
        return_time = instance['distance_matrix'][customer_id][0]
        updated_elapsed_time = elapsed_time + \
            instance['distance_matrix'][last_customer_id][customer_id] + service_time + return_time
        # Validate vehicle load and elapsed time
        if (updated_vehicle_load <= vehicle_capacity) and (updated_elapsed_time <= depart_due_time):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
            elapsed_time = updated_elapsed_time - return_time
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            elapsed_time = instance['distance_matrix'][0][customer_id] + service_time
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


def print_route(route, merge=False):
    '''gavrptw.core.print_route(route, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    total_cost = 0
    route = ind2route(individual, instance)
    total_cost = 0
    for sub_route in route:
        sub_route_time_cost = 0
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
            time_cost = wait_cost * max(instance[f'customer_{customer_id}']['ready_time'] - \
                arrival_time, 0) + delay_cost * max(arrival_time - \
                instance[f'customer_{customer_id}']['due_time'], 0)
            # Update sub-route time cost
            sub_route_time_cost = sub_route_time_cost + time_cost
            # Update elapsed time
            elapsed_time = arrival_time + instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + instance['distance_matrix'][last_customer_id][0]
        sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        sub_route_cost = sub_route_time_cost + sub_route_transport_cost
        # Update total cost
        total_cost = total_cost + sub_route_cost
    fitness = 1.0 / total_cost
    return (fitness, )


def cx_partially_matched(ind1, ind2):
    '''gavrptw.core.cx_partially_matched(ind1, ind2)'''
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
    rule1to2 = dict(rule1to2)
    ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + \
        [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2+1:]]
    ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + \
        [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2+1:]]
    return ind1, ind2


def mut_inverse_indexes(individual):
    '''gavrptw.core.mut_inverse_indexes(individual)'''
    start, stop = sorted(random.sample(range(len(individual)), 2))
    temp = individual[start:stop+1]
    temp.reverse()
    individual[start:stop+1] = temp
    return (individual, )

def predict_fitness(individuals, instance, unit_cost, init_cost, wait_cost, delay_cost):
    fitnesses = []
    for individual in individuals:
        # Convert float keys to integer keys
        individual = [int(round(i)) for i in individual]
        fitness = eval_vrptw(individual, instance, unit_cost, init_cost, wait_cost, delay_cost)[0]
        fitnesses.append(fitness)
    return fitnesses

def plot_routes(instance, best_ind, output_path):
    # Use ind2route to generate sub-routes for each vehicle
    routes = ind2route(best_ind, instance)

    # Extract customers from the instance based on keys starting with 'customer_'
    customers = {key: value for key, value in instance.items() if key.startswith('customer_')}

    # Extract depot details
    depot = instance['depart']

    # Extract coordinates for depot and customers
    coordinates = []
    labels = []

    # Add depot
    coordinates.append((depot['coordinates']['x'], depot['coordinates']['y']))
    labels.append('Depot')

    # Map customer indices to coordinates
    customer_indices = {}
    for index, (key, value) in enumerate(customers.items(), start=1):  # Start from 1 assuming depot is 0
        coordinates.append((value['coordinates']['x'], value['coordinates']['y']))
        labels.append(key.split('_')[1])  # Extract only the numeric part of the customer key
        customer_indices[index] = len(coordinates) - 1  # Map index to coordinates

    # Increase the plot size for better visibility (Change size to 24x18 inches)
    plt.figure(figsize=(24, 18))  # Increased the size to 24x18 inches

    # Plot depot
    plt.scatter(*zip(*coordinates[:1]), color='red', label='Depot', s=200, zorder=5)

    # Plot customers
    plt.scatter(*zip(*coordinates[1:]), color='blue', label='Customers', s=100, zorder=5)

    # Define a custom color palette
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Use itertools.cycle to repeat the colors if the number of routes exceeds the number of custom colors
    color_cycle = itertools.cycle(custom_colors)

    # Plot each vehicle's route using sub-routes from ind2route
    for i, route in enumerate(routes):
        route_coords = [coordinates[0]]  # Start at the depot (index 0)
        for customer_id in route:
            if customer_id not in customer_indices:
                print(f"Error: Customer ID {customer_id} not found in customer_indices.")
                continue
            route_coords.append(coordinates[customer_indices[customer_id]])
        route_coords.append(coordinates[0])  # End back at the depot

        # Get the next color from the cycle
        color = next(color_cycle)

        # Draw lines connecting the points along the route with transparency
        plt.plot(*zip(*route_coords), marker='o', color=color, linestyle='-', linewidth=3,
                 label=f'Vehicle {i + 1}', alpha=0.8, zorder=3)

    # Add labels for the depot and customers with offsets to avoid overlapping
    label_offset_x, label_offset_y = 1.5, 1.5  # Adjust the label offset as necessary
    for i, label in enumerate(labels):
        plt.text(coordinates[i][0] + label_offset_x, coordinates[i][1] + label_offset_y,
                 label, fontsize=12, ha='right', zorder=6)  # Increased font size for better visibility

    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.title('Vehicle Routes with Customer Connections', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Set the DPI (resolution) of the output image for better quality
    route_image_filename = 'route_image.png'
    route_image_path = os.path.join(output_path, route_image_filename)

    # Save the plot with higher DPI (e.g., 300 for high resolution)
    plt.savefig(route_image_path, dpi=300)
    plt.close()  # Close the plot to avoid display if not needed

    # Return the path of the saved image
    return route_image_path

def generate_lime_explanation(model, X_np, best_ind_np, ind_size, output_path, feature_names):
    """
    Generates LIME explanations and saves the explanation plot as an image and HTML file.
    """

    # Check if best_ind_np is a single sample and reshape if necessary
    if len(best_ind_np.shape) == 1:
        print(f"Reshaping best_ind_np from {best_ind_np.shape} to (1, {best_ind_np.shape[0]})")
        best_ind_np = best_ind_np.reshape(1, -1)

    # Ensure feature names match the number of columns in X_np
    if X_np.shape[1] != len(feature_names):
        raise ValueError(f"Mismatch: feature_names has {len(feature_names)} elements, but X_np has {X_np.shape[1]} features")

    # LIME expects 2D data, so ensure best_ind_np is in the right shape (1 sample, N features)
    if best_ind_np.shape[1] != X_np.shape[1]:
        raise ValueError(f"Mismatch: best_ind_np has {best_ind_np.shape[1]} features, but X_np has {X_np.shape[1]} features")

    # Initialize the LIME explainer for tabular data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_np,  # Training data used to fit the explainer
        feature_names=feature_names,
        class_names=['Predicted Outcome'],  # Adjust based on your model
        verbose=True,
        mode='regression'  # or 'classification' depending on your model type
    )

    # Pick the first sample from best_ind_np to explain (since LIME explains one instance at a time)
    sample_to_explain = best_ind_np[0]

    # Generate explanation for the selected sample
    print(f"Generating LIME explanation for input with shape: {best_ind_np.shape}")
    explanation = explainer.explain_instance(
        sample_to_explain,
        model.predict,  # Prediction function
        num_features=len(feature_names)  # Number of features to show in the explanation
    )

    # Save the LIME explanation plot as an image
    explanation.save_to_file(f'{output_path}/lime_explanation.html')  # Save interactive HTML

    # Adjust the plot to ensure labels are visible
    plt.figure()
    fig = explanation.as_pyplot_figure()  # Generate the plot

    # Adjust margins to ensure the left side labels are fully visible
    fig.subplots_adjust(left=0.3)  # Increase the left margin to 0.3 (adjust this value if needed)

    lime_plot_path = f"{output_path}/lime_explanation_plot.png"
    plt.savefig(lime_plot_path, bbox_inches='tight')  # Use bbox_inches='tight' to ensure nothing is cut off
    plt.close()

    # Return paths to the saved files
    return lime_plot_path


def generate_shap_explanation(model, X_np, best_ind_np, ind_size, output_path, feature_names):
    """
    Generates SHAP explanations and saves the summary plot as an image.
    """
    # Initialize SHAP Explainer for your model
    explainer = shap.Explainer(model, X_np)

    # Check if best_ind_np is a single sample and reshape if necessary
    if len(best_ind_np.shape) == 1:
        print(f"Reshaping best_ind_np from {best_ind_np.shape} to (1, {best_ind_np.shape[0]})")
        best_ind_np = best_ind_np.reshape(1, -1)

    # Generate SHAP values for the best individual
    print(f"Generating SHAP values for input with shape: {best_ind_np.shape}")
    shap_values = explainer(best_ind_np)

    # Ensure feature names match X_np's features
    if X_np.shape[1] != len(feature_names):
        raise ValueError(f"Mismatch: feature_names has {len(feature_names)} elements, but X_np has {X_np.shape[1]} features")

    # SHAP expects the number of rows in `X_np` to match the number of SHAP values' rows
    assert X_np.shape[1] == best_ind_np.shape[1], \
        f"Feature mismatch: X_np has {X_np.shape[1]} features but best_ind_np has {best_ind_np.shape[1]} features!"

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, best_ind_np, feature_names=feature_names)

    # Save the SHAP summary plot as an image
    shap_summary_plot_path = f"{output_path}/shap_summary_plot.png"
    plt.savefig(shap_summary_plot_path)
    plt.close()

    # Return paths to the saved files
    return shap_summary_plot_path


def extract_features(ind, instance):
    """
    Extract meaningful features from an individual and the problem instance.
    """
    # Features related to the customers
    total_demand = sum([instance['customer_' + str(i)]['demand'] for i in ind])  # Sum of demands
    total_service_time = sum([instance['customer_' + str(i)]['service_time'] for i in ind])  # Total service time
    total_ready_time = sum([instance['customer_' + str(i)]['ready_time'] for i in ind])  # Sum of ready times
    total_due_time = sum([instance['customer_' + str(i)]['due_time'] for i in ind])  # Sum of due times

    # Route-level features using the distance matrix
    total_route_distance = sum([instance['distance_matrix'][i][j] for i, j in zip(ind[:-1], ind[1:])])  # Total distance
    avg_distance_per_customer = total_route_distance / (len(ind) - 1)  # Average distance
    max_distance = max([instance['distance_matrix'][i][j] for i, j in zip(ind[:-1], ind[1:])])  # Max distance
    min_distance = min([instance['distance_matrix'][i][j] for i, j in zip(ind[:-1], ind[1:])])  # Min distance

    # Calculate violations (e.g., late arrivals beyond due_time)
    time_window_violations = sum([1 for i in ind if instance['customer_' + str(i)]['ready_time'] > instance['customer_' + str(i)]['due_time']])

    # Vehicle capacity
    vehicle_capacity = instance['vehicle_capacity']
    max_vehicle_number = instance['max_vehicle_number']

    # Return the feature list
    return [
        total_demand,               # Total demand on the route
        total_service_time,          # Total service time
        total_ready_time,            # Total ready time
        total_due_time,              # Total due time
        total_route_distance,        # Total distance traveled
        avg_distance_per_customer,   # Average distance between customers
        max_distance,                # Maximum distance between any two consecutive customers
        min_distance,                # Minimum distance between any two consecutive customers
        time_window_violations,      # Number of time window violations
        vehicle_capacity,            # Maximum vehicle capacity
        max_vehicle_number           # Maximum number of vehicles
    ]


def run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size,
                cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    '''Run the genetic algorithm for the VRPTW problem and generate LIME/SHAP explanations.'''

    if customize_data:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    else:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = load_instance(json_file=json_file)
    if instance is None:
        return

    # Initialize DEAP tools and operators
    creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', eval_vrptw, instance=instance, unit_cost=unit_cost,
                     init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_inverse_indexes)

    pop = toolbox.population(n=pop_size)
    csv_data = []
    print('Start of evolution')

    # Collect data for LIME and SHAP
    X = []
    y = []

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

        # Extract meaningful features instead of customer IDs
        features = extract_features(ind, instance)
        X.append(features)  # Collect features for model training
        y.append(fit[0])  # Collect fitness for model training

    # Convert collected data to numpy arrays
    X_np = np.array(X)  # Now this is an array of meaningful features, not customer IDs
    y_np = np.array(y)

    # Train a model for LIME and SHAP explanations
    model = RandomForestRegressor()
    model.fit(X_np, y_np)

    # **Extract feature names dynamically based on the extracted features**:
    # We will use the names from the `extract_features` function
    feature_names = [
        'Total Demand',
        'Total Service Time',
        'Total Ready Time',
        'Total Due Time',
        'Total Route Distance',
        'Avg Dist per Customer',
        'Max Distance',
        'Min Distance',
        'Time Window Violations',
        'Vehicle Capacity',
        'Max Vehicle Number'
    ]

    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)

    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, instance))
    print(f'Total cost: {1 / best_ind.fitness.values[0]}')

    # Save LIME and SHAP explanations
    output_path = os.path.join(BASE_DIR, 'results')
    best_ind_np = np.array([extract_features(best_ind, instance)])  # Use meaningful features for best individual

    # Save the final routes plot
    route_image_path = plot_routes(instance, best_ind, output_path)

    # **Generate SHAP and LIME explanations using feature names**
    shap_summary_plot_path = generate_shap_explanation(model, X_np, best_ind_np, ind_size, output_path, feature_names)
    lime_plot_path = generate_lime_explanation(model, X_np, best_ind_np, ind_size, output_path, feature_names)

    if export_csv:
        csv_file_name = f'{instance_name}_uC{unit_cost}_iC{init_cost}_wC{wait_cost}' \
                        f'_dC{delay_cost}_iS{ind_size}_pS{pop_size}_cP{cx_pb}_mP{mut_pb}_nG{n_gen}.csv'
        csv_file = os.path.join(BASE_DIR, 'results', csv_file_name)
        print(f'Write to file: {csv_file}')
        make_dirs_for_file(path=csv_file)
        if not exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', encoding='utf-8', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)

    return lime_plot_path, shap_summary_plot_path, route_image_path