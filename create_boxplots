# Create a figure and axes
fig, ax = plt.subplots()
for i in range(17):
  print(np.var(gp_mean[i]))
# Create the boxplot
gp_imp = np.array([gp_mean[1], gp_mean[3], gp_mean[4], gp_mean[15]])
print(np.squeeze(gp_imp).shape)
#gp_imp = np.reshape(gp_imp, (len(gp_imp), -1))
boxplot = ax.boxplot(np.transpose(np.squeeze(gp_imp)))
print(len(gp_imp))
# Set the labels for each boxplot
properties_full=['absorption coefficient b', 'latent heat of fusion', 'latent heat of vaporisation',  'liquid density']

ax.set_xticklabels(properties_full)

# Set the title and axis labels
ax.set_title('Multiple Boxplots')
ax.set_xlabel('Boxplot')
ax.set_ylabel('Values')

# Show the plot
plt.show()

colors = ['blue', 'black', 'red', 'green']

fig = go.Figure()

for xd, yd, cls in zip(properties_full, np.squeeze(gp_imp), colors):
        fig.add_trace(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker_size=2,
            line_width=1)
        )

fig.update_layout(
    title='Boxplots showing Cylindricity variation',
    yaxis_title='Depth',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False,
    font=dict(
        size=20  # Set the desired font size
    )
)
fig.show()
