import pybamm

models = [
    pybamm.lithium_ion.SPM(),  # single particle model
    pybamm.lithium_ion.SPMe(),  # single particle model w/ electrolyte
    pybamm.lithium_ion.DFN(),
]

sims = []

for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)

pybamm.dynamic_plot(sims)
