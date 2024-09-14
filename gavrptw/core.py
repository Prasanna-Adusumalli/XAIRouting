# -*- coding: utf-8 -*-

'''gavrptw/core.py'''

import os
import io
import random
from csv import DictWriter

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

    # Plot the routes
    plt.figure(figsize=(18, 12))  # Increase figure size for better spacing

    # Plot depot
    plt.scatter(*zip(*coordinates[:1]), color='red', label='Depot', s=150, zorder=5)

    # Plot customers
    plt.scatter(*zip(*coordinates[1:]), color='blue', label='Customers', s=70, zorder=5)

    # Assign a random color to each vehicle route
    colors = plt.cm.get_cmap('tab10', len(routes))  # Use 'tab10' colormap for distinct colors

    # Step 2: Plot each vehicle's route using sub-routes from ind2route
    for i, route in enumerate(routes):
        route_coords = [coordinates[0]]  # Start at the depot (index 0)
        for customer_id in route:
            if customer_id not in customer_indices:
                print(f"Error: Customer ID {customer_id} not found in customer_indices.")
                continue
            route_coords.append(coordinates[customer_indices[customer_id]])
        route_coords.append(coordinates[0])  # End back at the depot

        # Draw lines connecting the points along the route with transparency
        plt.plot(*zip(*route_coords), marker='o', color=colors(i), linestyle='-', linewidth=2,
                 label=f'Vehicle {i + 1}', alpha=0.7, zorder=3)

    # Add labels for the depot and customers with offsets to avoid overlapping
    label_offset_x, label_offset_y = 1.5, 1.5  # Adjust the label offset as necessary
    for i, label in enumerate(labels):
        plt.text(coordinates[i][0] + label_offset_x, coordinates[i][1] + label_offset_y,
                 label, fontsize=9, ha='right', zorder=6)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Finalised Vehicle Routes')
    plt.legend()
    plt.grid(True)

    route_image_filename = 'route_image.png'
    route_image_path = os.path.join(output_path, route_image_filename)

    # Save the plot
    plt.savefig(route_image_path)
    plt.close()  # Close the plot to avoid display if not needed

    # Return the path of the saved image
    return route_image_path

def generate_lime_explanation(X_np, best_ind_np, instance, unit_cost, init_cost, wait_cost, delay_cost, ind_size, output_path):
    '''Generate LIME explanation for the best individual'''
    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=[f'Customer {i}' for i in range(ind_size)],
        mode='regression'
    )

    # Define the prediction function for LIME
    def lime_predict_fn(X):
        return np.array(predict_fitness([list(x) for x in X], instance, unit_cost, init_cost, wait_cost, delay_cost))

    # Get LIME explanation for the best individual
    explanation = explainer.explain_instance(best_ind_np[0], lime_predict_fn)

    # Save LIME explanation to HTML
    explanation_html = explanation.as_html()
    explanation_html_file_path = os.path.join(output_path, 'lime_explanation.html')
    with open(explanation_html_file_path, 'w', encoding='utf-8') as f:
        f.write(explanation_html)
    print(f'LIME explanation saved to {explanation_html_file_path}')

    return explanation_html_file_path


def generate_shap_explanation(model, X_np, best_ind_np, ind_size, output_path):
    '''Generate SHAP explanations for the best individual and the entire dataset'''
    shap_explainer = shap.TreeExplainer(model)

    # Local SHAP explanation for the best individual
    shap_values_best = shap_explainer.shap_values(best_ind_np)

    # Save SHAP force plot as HTML
    shap_html_file_path = os.path.join(output_path, 'shap_explanation.html')
    shap.force_plot(shap_explainer.expected_value, shap_values_best, best_ind_np,
                    feature_names=[f'Customer {i}' for i in range(ind_size)])

    # Save SHAP force plot to HTML
    shap.save_html(shap_html_file_path, shap.force_plot(
        shap_explainer.expected_value,
        shap_values_best,
        best_ind_np,
        feature_names=[f'Customer {i}' for i in range(ind_size)]
    ))
    print(f'SHAP explanation saved to {shap_html_file_path}')

    # Global SHAP feature importance plot
    shap_values = shap_explainer.shap_values(X_np)

    # Plot and save global SHAP feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_np, feature_names=[f'Customer {i}' for i in range(ind_size)])
    shap_summary_plot_path = os.path.join(output_path, 'shap_summary_plot.png')
    plt.savefig(shap_summary_plot_path)
    plt.close()
    print(f'SHAP summary plot saved to {shap_summary_plot_path}')
    return shap_html_file_path, shap_summary_plot_path


def run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
                cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    '''gavrptw.core.run_gavrptw(instance_name, unit_cost, init_cost, wait_cost, delay_cost,
        ind_size, pop_size, cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False)'''

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
    toolbox.register('evaluate', eval_vrptw, instance=instance, unit_cost=unit_cost, \
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
        X.append(ind)  # Collect individuals for model training
        y.append(fit[0])  # Collect fitness for model training

    # Convert collected data to numpy arrays
    X_np = np.array([list(ind) for ind in X])
    y_np = np.array(y)

    # Train a model for LIME and SHAP explanations
    model = RandomForestRegressor()
    model.fit(X_np, y_np)

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
    best_ind_np = np.array([list(best_ind)])

    # Save the final routes plot
    route_image_path = plot_routes(instance, best_ind, output_path)

    shap_html_path, shap_summary_plot_path=generate_shap_explanation(model, X_np, best_ind_np, ind_size, output_path)
    lime_html_path =generate_lime_explanation(X_np, best_ind_np, instance, unit_cost, init_cost, wait_cost, delay_cost, ind_size, output_path)

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
    return lime_html_path, shap_html_path, shap_summary_plot_path, route_image_path