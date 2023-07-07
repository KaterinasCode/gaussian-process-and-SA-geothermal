import plotly.graph_objects as go

s_imp = []
st_imp = []
params_imp = []
properties_full=['absorption coefficient a','absorption coefficient b', 'absorption coefficient c', 'latent heat of melt', 'latent heat of vaporisation', 'conductivity parameter 1','conductivity parameter 3', 'conductivity parameter 3', 'specific heat parameter 1', 'specific heat parameter 2','melt temperature', 'vaporisation temperature', 'solid enthalpy', 'density parameter 1', 'density parameter 2', 'liquid density', 'solid density']
print(len(properties_full))
for i in range(len(s)):
  if (s_1[i]>50e-4 or st_1[i]>50e-4):
      s_imp.append(s[i])
      st_imp.append(st[i])
      params_imp.append(properties_full[i])
      print(i)

fig = go.Figure(data=[
    go.Bar(name='First Order', x=params_imp, y=s_imp, marker_color = 'fuchsia'),
    go.Bar(name='Total Order', x=params_imp, y=st_imp, marker_color = 'blue')
])
# Change the bar mode
fig.update_layout(barmode='group', title='First and Total order indices, QoI: Depth',)
fig.show()
