# %%
# The components and pouch cell are defined by a set of equations
# The solver goes through them all one-by-one and checks if there is a single unknown variable that can be solved for.
# If so, it solves for it and updates the attributes of the component.
# It iterates through all equations until there are no more unknown variables that can be solved for.

# This makes the solver able to calculate attributes both
# in a top-down manner (starting with attributes of the cell and solving for attributes of the components)
# and in a bottom-up manner (starting with attributes of the components and solving for attributes of the cell).

# However, the solver is currently limited to 1 unknown variable per equation.
# This means that it is not fully able to calculate top-down unless equations and attributes result in a chain of 1-unknown/1-equation.

# To adress this, the package also includes a brute-force tuning function that uses scipy.optimize
# Here we show how to use the tuning function.

# import the example cell
from Battery_Pack_Solver.cells.example_cell import pouch_cell_attrs

# import the PouchCell class
from Battery_Pack_Solver.component import PouchCell

# The example cell can be fully calculated without inconsistencies, so we can directly initialize the cell.
pc = PouchCell(pouch_cell_attrs, verbose="v")

# we print the cell summaries
pc.print(set="M")
pc.print(set="L")
pc.print(set="D")

# %%

# Now, say we want to know how thick the cathode needs to be in order for the total cell mass to be 20 g.
# we remove the input for cathode film thickness so that it can be solved for
del pouch_cell_attrs["components"]["c"]["Lf"]  # cm, thickness of cathode film
pouch_cell_attrs["M"] = 20  # g, total mass of the cell


# rerun the calculation with the updated input attributes
pc = PouchCell(pouch_cell_attrs, verbose="v")
# print the cell summaries again to see the updated cathode film thickness and the resulting mass breakdown
pc.print(set="M")
pc.print(set="L")

# although the system is uniquely solvable,
# the solver is not able to calculate further than 1 unknown variable at a time
#
# This is addressed with a bottom-up tuning function that uses scipy.optimize

# %%


# we remove the target attribute and the tuning attribute
# and replace them with the tuning dictionary that specifies
# the tuning variable,
# the optimization variable,
# the initial guess for the tuning variable,
# and the target value for the optimization variable
del pouch_cell_attrs["components"]["c"]["Lf"]  # cm, thickness of cathode film
del pouch_cell_attrs["M"]  # g, total mass of the cell
pouch_cell_attrs["tune"] = {
    # Will be tuned until the optimal value is found
    "tuning_variable": ("c", "Lf"),
    # The variable that is targeted for optimization
    "optimization_variable": ("P", "M"),
    # initial guess for cathode film thickness
    "guess": 50e-4,
    # Target total mass of the cell
    "target": 20,
}

pc = PouchCell(pouch_cell_attrs, verbose="v")
# print the cell summaries again to see the updated cathode film thickness and the resulting mass breakdown
pc.print(set="M")
pc.print(set="L")

# now we see that the cathode film thickness is 76um and the total mass of the cell is 20 g as desired.
