# %%

# Being written in python, the cell configration can be looped programattically to sweep through different design parameters.

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

# Say we want to see how the cell mass changes with cathode film thickness. We can loop through different values of cathode film thickness and calculate the resulting cell mass.

ll = [
    i * 1e-4 for i in range(10, 240, 20)
]  # cm, different cathode film thicknesses to sweep through
Ms = []  # list to store the resulting cell masses
ls = []
for L in ll:
    pouch_cell_attrs["components"]["c"]["Lf"] = L  # update the cathode film thickness
    pc = PouchCell(
        pouch_cell_attrs
    )  # recalculate the cell with the updated cathode film thickness
    Ms.append(pc.M)  # store the resulting cell mass
    ls.append(pc.c.Lf)  # store the resulting cell mass

print(f"\n{'Thickness (um)':<20} {'Mass (g)':<10}")
print("-" * 30)
for L, M in zip(ls, Ms):
    print(f"{L * 1e4:<20.0f} {M:<10.4f}")
