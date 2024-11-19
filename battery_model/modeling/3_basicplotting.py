import pybamm

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])

#!TODO getting all variable names
# print(model.variable_names())

#!TODO searching for variable names
# print(model.variables.search("electrolyte"))

# use the variable names to selectively plot
output_variables = [
    ["Electrode current density [A.m-2]", "Electrolyte current density [A.m-2]"],
    "Voltage [V]",
]
# sim.plot(output_variables=output_variables)

#!TODO produce voltage plot showing contribution of overpotentials
sim.plot_voltage_components()
