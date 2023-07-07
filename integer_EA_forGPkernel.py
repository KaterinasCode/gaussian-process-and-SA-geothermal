import random
from deap import base, creator, tools
import gpflow
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
k1=gpflow.kernels.Matern12()
k2=gpflow.kernels.Matern32()
k3=gpflow.kernels.Matern52()
k4 = gpflow.kernels.RBF()
k5=gpflow.kernels.Linear()
k6=gpflow.kernels.Constant()
k7=gpflow.kernels.Cosine()
k8 = gpflow.kernels.White()
kernels = [k1,k2,k3,k4,k5,k1*k1, k2*k5, k4*k5, k7]
X = sample_s
individuals_list = []
fitness_list = []

# Create the toolbox
toolbox = base.Toolbox()

# Define the optimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Register the attribute generator
#toolbox.register("attribute", random.randint, 0, 1)  # Generate 0 or 1 for each attribute

# Register the individual creation operator
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=9)

# Register the population creation operator
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
def mutate_bitflip(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = abs(individual[i]-1)
    return individual,

def evaluate(individual):
    print(individual)
    for i in range(len(individuals_list)):
        if (individual == individuals_list[i]):
            return(fitness_list[i])
    individuals_list.append(individual)
    k=3
    kf = KFold(n_splits=k)
    errors = []
    lmls = []
    f = 0
    first_nonnegative_index = None
    for i, value in enumerate(individual):
        if individual[i] >= 0.5:
            first_nonnegative_index = i
            print(first_nonnegative_index)
            break
    if (first_nonnegative_index is None):
        kernel = k6
    else:
        kernel = kernels[first_nonnegative_index]
    for i in range(len(individual)):
        if (individual[i]>=0.5):
            kernel +=kernels[i]

    for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for the current fold
        X_train, X_test = sample_s[train_index], sample_s[test_index]
        y_train, y_test = obj2[train_index], obj2[test_index]
        data = (X_train.reshape(-1, 17), y_train.reshape(-1, 1))
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
            return [-2000]
        error = mean_squared_error(y_test, y_pred)
        lml = model.log_marginal_likelihood().numpy()
        print(lml)
        errors.append(error)
        if lml >1e6:
            return 0
        else:
            lmls.append(lml)
    print(np.mean(errors), np.mean(lmls))
    return (-np.mean(lmls),)

# Register the evaluation function
toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate_bitflip, indpb=0.1)  # indpb is the probability of flipping each bit
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)  # Crossover operator
# Define the evolutionary algorithm parameters

population_size = 9
generations = 3

# Create an initial population
population = []
#population = toolbox.population(n=population_size)
best_fitness_values = []
for _ in range(1):
    for i in range(population_size):
        individual = creator.Individual()
        individual_in = np.zeros(9)
        individual_in[i]=1
        individual.extend(individual_in)
        population.append(individual)
    # Generate random values for each decision variable
# Perform the evolution
for generation in range(generations):
    offspring = [toolbox.clone(ind) for ind in population]
    if (generation>=1):
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
