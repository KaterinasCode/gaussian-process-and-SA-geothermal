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
kernels = [k1,k2,k3,k4,k5,k7]
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

    k = 3
    kf = KFold(n_splits=k)
    lmls = []

    active_kernels = [kernels[i] for i, value in enumerate(individual[:-1]) if value >= 0.5]

    if len(active_kernels) == 0:
        return (0),
    firstnonnegativeindex = None
    for i in range(len(individual)-1):
        if individual[i]>0.5: 
            firstnonnegativeindex = i
    kernel = active_kernels[0]
    for i in range(firstnonnegativeindex,len(individual)-1):
        if (individual[i]>=0.5 and individual[len(individual)-1] <=0.5 ):
            kernel +=kernels[i]
        elif (individual[i]>=0.5 and individual[len(individual)-1] >0.5 ):
            kernel *=kernels[i]    

    print(individual)
    print("Kernel Components:")

    if isinstance(kernel, gpflow.kernels.Composite):
        for component in kernel.kernels:
            print(component)
    else:
        print(kernel)

    for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for the current fold
        X_train, X_test = sample_s[train_index], sample_s[test_index]
        y_train, y_test = obj1[train_index], obj1[test_index]
        data = (X_train.reshape(-1, 15), y_train.reshape(-1, 1))
        model = gpflow.models.GPR(data, kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(model.training_loss, variables=model.trainable_variables)
        y_pred = model.predict_y(X_test)[0]
        has_nan = np.isnan(y_pred)
        if(np.any(has_nan)==True):
            return [-2000]
        lml = model.log_marginal_likelihood().numpy()
        if lml >1e6:
            return (0,)
        else:
            lmls.append(lml)
    print(np.mean(lmls))
    return (-np.mean(lmls),)

# Register the evaluation function
toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate_bitflip, indpb=0.1)  # indpb is the probability of flipping each bit
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)  # Crossover operator
# Define the evolutionary algorithm parameters

population_size = 6
generations = 3

# Create an initial population
population = []
#population = toolbox.population(n=population_size)
best_fitness_values = []
for _ in range(1):
    for i in range(population_size):
        individual = creator.Individual()
        individual_in = np.zeros(7)
        individual_in[i]=1
        if i%2==0:
            individual_in[len(individual_in)-1] =1 
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
plt.figure()
for i in range(len(fitness_list)):
    plt.scatter(i, fitness_list[i])

plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness Progression')
plt.show()
