import pybamm
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)

sim.solve([0,3600])
sim.plot()
