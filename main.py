# Battery_Pack_Solver/main.py
from Modules.Battery_Pack_Solver.component import PouchCell

# Define attributes for components (anode, cathode, etc.)
cu_foil_attrs = {
    "Description": "Cu-foil",
    "desc": "a_cc",
    "por": 0.0,  # porosity. 0 means solid metal, 1 means no solid phase
    "Lf": 9e-4,  # cm, thickness of film
    "Ma": 4.77e-3,  # g/cm2, Areal mass Cu current collector
    "Dm": 9.09,  # $/kq, Material cost of Copper foil
}

al_foil_attrs = {
    "Description": "Al-foil",
    "desc": "c_cc",
    "por": 0.0,  # porosity. 0 means solid metal, 1 means no solid phase
    "Lf": 10e-4,  # cm, thickness of film
    "Ma": 2.84e-3,  # g/cm2, Areal mass of Al current collector
    "Dm": 6.00,  # $/kq, Material cost of Aluminlum foi
}

separator_attrs = {
    "Description": "Separator",
    "desc": "s",
    "d0": 3.6,  # cm, width of film
    "d1": 5.6,  # cm, length of film
    "Lf": 20e-4,  # cm, thickness of separator
    "Mv": 0.946,  # g/cm3, density of separator
    "por": 0.5,  # 1, volume liquid phase per volume separator
    "Dm": 79.71,  # $/kq, Material cost of separator
}

packaging_attrs = {
    "Description": "Packaging",
    "desc": "pg",
    "por": 0.0,  # porosity. 0 means solid metal, 1 means no solid phase
    "d0": 3.6,  # cm, width of film
    "d1": 5.6,  # cm, length of film
    "Lf": 230e-4,  # cm, thickness of packaging foil (there are 2 layers per pouch cell)
    "M": 2.5248,  # g, total mass of packaging foil
    "Dm": 0.38,  # $/kq, Material cost of packaging foil
}

electrolyte_attrs = {
    "Description": "Electrolyte",
    "desc": "el",
    "Mv": 1.2,  # g/cm3, density of electrolyte
    "Dm": 15.00,  # $/kq, Material cost
}
cathode_attrs = {
    "Description": "Cathode",
    "desc": "c",
    "Lf": 50e-4,  # cm, thickness of film
    "por": 0.3,  # 1, volume non-solid phase per volume electrode
    "Qm_am": 180,  # mAh/g
    "Vavg": 3.7,  # V, Average Voltage of Electrode
    # 1, mass ratios of components (am, binder, carbon)
    "w_i": [0.96, 0.02, 0.02],
    # g/cm3, mass per volume of components (am, binder, carbon)
    "Mv_i": [4.75, 1.8, 2.26],
    # $/kq, Material cost of components (am, binder, carbon)
    "Dm_i": [8.541, 18.00, 7.15],
}
anode_attrs = {
    "Description": "Anode",
    "desc": "a",
    "Qm_am": 3860,  # mAh/g
    "por": 0.0,  # porosity. 0 means solid metal, 1 means no solid phase
    "Vavg": 0.0,  # V, Average Voltage of Electrode, this is not used in the anode
    # 1, mass ratios of components (am, binder, carbon)
    "w_i": [1.00],
    # g/cm3, mass per volume of components (am, binder, carbon)
    "Mv_i": [0.534],
    # $/kq, Material cost of components (am, binder, carbon)
    "Dm_i": [49.60],
}

pouch_cell_attrs = {
    "Description": "Pouch Cell",
    "desc": "pc",
    "N": 10,  # number of layers. defined as number of double sided anodes.
    "d0_a": 3.6,  # cm, width of anode
    "d1_a": 5.6,  # cm, length of anode
    "d0_c": 3.6,  # cm, width of cathode
    "d1_c": 5.6,  # cm, length of cathode
    "EC": 3e-3,  # g/mAh, E/C Ratio, mass electrolyte per capacity
    "NP": 1.1,  # 1, N/P Ratio
    "components": {
        "el": electrolyte_attrs,
        "c": cathode_attrs,
        "c_cc": al_foil_attrs,
        "a": anode_attrs,
        "a_cc": cu_foil_attrs,
        "s": separator_attrs,
        "pg": packaging_attrs,
    },
    "tune": {
        # Will be tuned until the optimal value is found
        "tuning_variable": ("c", "Lf"),
        # The variable that is targeted for optimization
        "optimization_variable": ("P", "M"),
        # initial guess for cathode film thickness
        "guess": 50e-4,
        # Target total mass of the cell
        "target": 15,
    },
}

pc = PouchCell(pouch_cell_attrs, verbose="v")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# pc.pg.print_attr_eq()
# pc.c_cc.print_attr_eq()
# pc.a_cc.print_attr_eq()
# pc.s.print_attr_eq()
# pc.el.print_attr_eq()
# pc.c.print_attr_eq()
# pc.a.print_attr_eq()
# pc.print_attr_eq()
pc.c.solve_component(verbose="vvv")


# pc.print(set="libatt")
pc.print(set="M")
pc.print(set="L")
pc.print(set="D")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# pc.pg.clear_and_reinitialize()
# pc.pg.solve_component(verbose="v")
# pc.pg.print_attr_eq()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
