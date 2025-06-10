# Battery_Pack_Solver/Xiao_2022_Power_Pouch.py
#
#
# Xiao, J., Anderson, C., Cao, X., Chang, H.-J., Feng, R.,
# Huang, Q., Jin, Y., Job, H., Kim, J.-M., Le, P. M. L., Liu,
# D., Seymour, L., Shamim, N., Shi, L., & Sivakumar, B. (2022).
#   Perspective—Electrochemistry in understanding
#   and designing electrochemical energy storage systems.
#   Journal of the Electrochemical Society, 169(1), 10524.
#   https://doi.org/10.1149/1945-7111/ac4a55
from Modules.Battery_Pack_Solver.component import PouchCell

# NMC 333 Cathode
cathode_attrs = {
    "Description": "Cathode",
    "desc": "c",
    "Qm_am": 140,  # mAh/g
    "w_i": [0.97],  # active mass loading
    "Ma": 15e-3,  # g/cm2, areal mass of cathode film
    "Qa": 2.0,  # mAh/cm2, areal film capacity
    "Mv": 3.3,  # g/cm3, density of cathode
    "Lf": 45e-4,  # cm, thickness of film
    "d0": 8.9,  # cm, width of film
    "d1": 26.4,  # cm, length of film
    "Vavg": 3.6,  # V, Average Voltage of Electrode
    "Q": 60e3,  # mAh, Total capacity of the cell
}
al_foil_attrs = {
    "Description": "Al-foil",
    "desc": "c_cc",
    "por": 0.0,  # porosity. 0 means solid
    "Lf": 10e-4,  # cm, thickness of aluminum current collector
    "d0": 8.9,  # cm, width of film
    "d1": 26.4,  # cm, length of film
    # Not from cited paper:
    "Mv": 2.7,  # g/cm3, Density of Al
}
# Graphite Anode
anode_attrs = {
    "Description": "Anode",
    "desc": "a",
    "Qm_am": 340,  # mAh/g
    "w_i": [0.96],  # active mass loading
    "Ma": 6.9e-3,  # g/cm2, areal mass of anode film
    "d0": 9.1,  # cm, width of film
    "d1": 26.6,  # cm, length of film
    # Not from cited paper:
    "Mv": 0.534,  # g/cm3, Density of Lithium
}
cu_foil_attrs = {
    "Description": "Cu-foil",
    "desc": "a_cc",
    "por": 0.0,  # porosity. 0 means solid
    "Lf": 6e-4,  # cm, thickness of copper current collector
    "d0": 9.1,  # cm, width of film
    "d1": 26.6,  # cm, length of film
    # Not from cited paper:
    "Mv": 8.96,  # g/cm3, Density of Cu
}
separator_attrs = {
    "Description": "Separator",
    "desc": "s",
    "Lf": 20e-4,  # cm, thickness of separator
    "d0": 9.1,  # cm, width of film
    "d1": 26.6,  # cm, length of film
    # Not from cited paper:
    "Mv": 0.946,  # g/cm3, density of separator
}

packaging_attrs = {
    "Description": "Packaging",
    "desc": "pg",
    "por": 0.0,  # porosity. 0 means solid
    "d0": 10.0,  # cm, width of film
    "d1": 33.0,  # cm, length of film
    "Lf": 110e-4,  # cm, thickness of packaging foil (there are 2 layers per pouch cell)
    # Not from cited paper:
    "M": 2.5,  # g, total mass of packaging foil
}

electrolyte_attrs = {
    "Description": "Electrolyte",
    "desc": "el",
    # Not from cited paper:
    "Mv": 1.2,  # g/cm3, density of electrolyte
}

pouch_cell_attrs = {
    "Description": "Pouch Cell",
    "desc": "pc",
    "N": 64,  # number of layers. defined as number of double sided anodes.
    "EC": 3e-3,  # g/mAh, E/C Ratio, mass electrolyte per capacity
    "NP": 1.1,  # 1, N/P Ratio
    "Em": 240,  # mWh/g, Energy Density of Electrode Materials
    "components": {
        "el": electrolyte_attrs,
        "c": cathode_attrs,
        "c_cc": al_foil_attrs,
        "a": anode_attrs,
        "a_cc": cu_foil_attrs,
        "s": separator_attrs,
        "pg": packaging_attrs,
    },
}

pc = PouchCell(pouch_cell_attrs, verbose="v")


pc.print(set="M")
pc.print(set="L")

pc.get_inconsistent()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
