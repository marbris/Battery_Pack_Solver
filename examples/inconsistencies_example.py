# import attributes from from the paper
# Xiao, J. et. al. (2022).
#   Perspective—Electrochemistry in understanding
#   and designing electrochemical energy storage systems.
#   Journal of the Electrochemical Society, 169(1), 10524.
#   https://doi.org/10.1149/1945-7111/ac4a55
from Battery_Pack_Solver.cells.Xiao_2022_Energy_Pouch import pouch_cell_attrs

# import the PouchCell class
from Battery_Pack_Solver.component import PouchCell

# PouchCell class uses the input attributes and
# the algebraic relationships in the PouchCell class that define the components and the pouch cell
# to calculate the remaining attributes of the cell and its components.
# It also checks for inconsistencies derived from the input attributes.
# verbose="v" will print each step of the calculation and any inconsistencies.
#   yellow: the attribute being solved for
#   green: input attributes
#   blue: derived attributes
pc = PouchCell(pouch_cell_attrs, verbose="v")

# %%
# given that these input attributes are rounded,
#   inconsistencies are expected.
# get_inconsistent() returns a list of
# all inconsistent equations,
# their right-hand side (RHS) and left-hand side (LHS) values,
# and the relative/absolute difference between them.
pc.get_inconsistent()

# In this case, all inconsistencies are below 5% relative error.
# For instance,
#   the areal capacity of the cathode film (c.Qa = 4.1 mAh/cm2)
#   the number of films (pc.N = 70),
#   and the total cathode capacity (c.Q = 132 Ah) are given as input.
# These don't have to be resolved, but the output from print() function will be ambigouos.

# %%

# These inconsistencies can be resolved by removing any offending input
# Here we remove the inputs that are more difficult to measure directly.
del pouch_cell_attrs["components"]["c"]["Qa"]  # mAh/cm2, areal capacity of cathode film
del pouch_cell_attrs["components"]["c"]["Mv"]  # g/cm3, density of cathode film
del pouch_cell_attrs["NP"]  # mAh_anode/mAh_cathode, NP ratio of the cell
# rerun calulation with the updated input attributes
pc = PouchCell(pouch_cell_attrs, verbose="v")
# no more inconsistencies
pc.get_inconsistent()

# %%

# print() prints various summaries of the cell.
#   Masses
pc.print(set="M")
#   Thicknesses
pc.print(set="L")

# %%

# print_attributes() prints all attributes of the component.
#   green: input attributes,
#   blue: derived attributes,
#   red: unsolved attributes.
# pc.pg.print_attributes()  # packaging attributes
# pc.c_cc.print_attributes() # cathode current collector attributes
# pc.a_cc.print_attributes() # anode current collector attributes
# pc.s.print_attributes() # separator attributes
# pc.el.print_attributes() # electrolyte attributes
# pc.c.print_attributes() # cathode attributes
# pc.a.print_attributes() # anode attributes
pc.print_attributes()  # all attributes of the cell and its components


# %%

# print_equations() prints all equations of the component.
#   green: input attributes,
#   blue: derived attributes,
#   red: unsolved attributes.
# pc.pg.print_equations()  # packaging attributes
# pc.c_cc.print_equations() # cathode current collector attributes
# pc.a_cc.print_equations() # anode current collector attributes
# pc.s.print_equations() # separator attributes
# pc.el.print_equations() # electrolyte attributes
# pc.c.print_equations() # cathode attributes
# pc.a.print_equations() # anode attributes
pc.print_equations()  # all attributes of the cell and its components


# %%

# print_attr_eq() prints both the attributes and the equations of the component.
# pc.pg.print_attr_eq()
# pc.c_cc.print_attr_eq()
# pc.a_cc.print_attr_eq()
# pc.s.print_attr_eq()
# pc.el.print_attr_eq()
# pc.c.print_attr_eq()
# pc.a.print_attr_eq()
pc.print_attr_eq()
