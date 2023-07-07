#Lengthscales EA
import random
from deap import base, creator, tools
fitness_list = []
best_fitness_values= []
ker = [1, 0, 1, 1, 0, 0, 1, 0, 1]#obj2
k1=gpflow.kernels.Matern12()
k3=gpflow.kernels.Matern52()
def evaluate(individual):
    #individuals_list.append(individual)
    k=3
    kf = KFold(n_splits=k)
    errors = []
    lengthscale = individual
    print(lengthscale)
    for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for the current fold
        X_train, X_test = sample_s[train_index], sample_s[test_index]
        y_train, y_test = obj2[train_index], obj2[test_index]
        data = (X_train.reshape(-1, 17), y_train.reshape(-1, 1))
        kernel = gpflow.kernels.Matern52(lengthscales = lengthscale) + gpflow.kernels.Matern12(lengthscales = lengthscale)
        model = gpflow.models.GPR(data, kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        f_mean, f_var = model.predict_f(sample_test_s, full_cov=False)
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(model.training_loss, variables=model.trainable_variables)
    # Make predictions on the test data
        y_pred = model.predict_y(X_test)[0]
        #plotmodel(X_test, y_test, y_pred)
    # Calculate the mean squared error for the current fold
        has_nan = np.isnan(y_pred)
        if(np.any(has_nan)==True):
            return [0.001]
        error = mean_squared_error(y_test, y_pred)
        errors.append(error)
    print(np.mean(errors))
    return (np.mean(errors),)
lower_bound = 0.0
upper_bound = 1.0
eta = 20.0  # Polynomial index, determines the degree of perturbation
indpb = 0.2  # Probability of mutating each individual gene
# Define the optimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, 0, 1)  # Variable range [0, 1]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=17)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta, low=lower_bound, up=upper_bound, indpb=indpb)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the evolutionary algorithm parameters
population_size = 10
generations = 10

# Create an initial population
population = toolbox.population(n=population_size)

# Perform the evolution
for generation in range(generations):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    # Apply mutation
    for mutant in offspring:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    # Evaluate the fitness of the new individuals
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        if (fit == nan):
            ind.fitness.values = 0.01
        else:
            ind.fitness.values = fit
            fitness_list.append(fit)

    # Select the next generation
    population[:] = toolbox.select(offspring, k=len(population))
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness_values.append(best_individual.fitness.values[0])


# Get the best individual and its fitness value
best_individual = tools.selBest(population, k=1)[0]
best_fitness = best_individual.fitness.values[0]

print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)
# Plot the objective function evolution
plt.plot(best_fitness_values)
plt.xlabel("Generation")
plt.ylabel("Objective Function")
plt.title("Evolution of Objective Function")
plt.show()
for i in range(len(fitness_list)):
    plt.scatter(i,fitness_list[i])
plt.xlabel("Generation")
plt.ylabel("Objective Function")
plt.title("Evolution of Objective Function")
plt.show()
