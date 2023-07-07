figsize = (12, 12)  # Width: 8 inches, Height: 6 inches

# Create the figure with the specified figsize
fig, ax = plt.subplots(figsize=figsize)
for i in range(len(fitness_list)):
    if (float(fitness_list[i][0])<1):
        plt.scatter(i,fitness_list[i], color = 'black')
    if (float(fitness_list[i][0]<-2550)):
        plt.scatter(i,fitness_list[i], color = 'red')

ax.set_title("Evolution of Objective Function", fontsize=24)

# Set the y-axis label with a larger font size
ax.set_ylabel("Negative Log Marginal Likelihood", fontsize=20)

# Set the x-axis label with a larger font size
ax.set_xlabel("Generation", fontsize=20)
plt.savefig("lml_evolution_obj2_kernel.png")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.show()
