# Battery_Pack_Solver

A Python tool for calculating the energy density of lithium-ion pouch cells.

There are many spreadsheet-based pouch cell calculators available in the literature. This tool does the same fundamental calculation, but is designed to be used programmatically, enabling things that are difficult or impossible in spreadsheets:

1. **Parameter sweeps** — Loop over any design variable to explore the design space.
2. **Universal solving** — Specify any combination of known parameters and solve for the rest. The calculation is not locked into a single input→output direction. For example, set a target cell mass and solve for the required cathode coating thickness.
3. **Transparent intermediate results** — Every calculation returns the full chain of intermediate values, making it easy to inspect where energy density is gained or lost and to sanity-check against published data.
4. **Resilient to inconsistent inputs** — Inconsistent inputs are flagged with detailed comparisons, allowing the input of proposed designs from literature where values are often rounded.

## What This Is Not

This is a **geometry and mass/energy bookkeeping tool**. It calculates energy density from cell design parameters — dimensions, densities, porosities, specific capacities, and voltages.

It does **not** model electrochemical behavior, rate capability, degradation, or cycle life. It answers the question *"given these design choices, what is the energy density of the resulting cell?"* — nothing more.

## Installation

```bash
pip install git+https://github.com/marbris/Battery_Pack_Solver.git
```

Or for development:

```bash
git clone https://github.com/marbris/Battery_Pack_Solver.git
cd Battery_Pack_Solver
pip install -e .
```

## Quick Start

### Define a cell and calculate its properties

```python
from Battery_Pack_Solver.cells.example_cell import pouch_cell_attrs
from Battery_Pack_Solver.component import PouchCell

pc = PouchCell(pouch_cell_attrs, verbose="v")
```

### Inspect the calculation breakdown

```python
pc.print(set="M")  # Mass breakdown
pc.print(set="L")  # Thickness breakdown
pc.print(set="D")  # Materials cost breakdown
```

### Sweep a design parameter

```python
from Battery_Pack_Solver.cells.example_cell import pouch_cell_attrs
from Battery_Pack_Solver.component import PouchCell

thicknesses = [i * 1e-4 for i in range(10, 240, 20)]
for L in thicknesses:
    pouch_cell_attrs["components"]["c"]["Lf"] = L
    pc = PouchCell(pouch_cell_attrs)
    print(f"Cathode thickness: {pc.c.Lf * 1e4:.0f} um, Cell mass: {pc.M:.4f} g")
```

### Solve for a target

What cathode thickness do I need for a total cell mass of 20 g?

```python
from Battery_Pack_Solver.cells.example_cell import pouch_cell_attrs
from Battery_Pack_Solver.component import PouchCell

pouch_cell_attrs["tune"] = {
    "tuning_variable": ("c", "Lf"),
    "optimization_variable": ("P", "M"),
    "guess": 50e-4,
    "target": 20,
}

pc = PouchCell(pouch_cell_attrs, verbose="v")
print(f"Required cathode thickness: {pc.c.Lf * 1e4:.0f} um")
```

## Examples

The example scripts in `examples/` are structured as cell-based Python scripts using `# %%` cell markers. They can be run:

- **Cell-by-cell** in any editor that supports the `# %%` format (VS Code, Neovim + IPython, Spyder, PyCharm, etc.)
- **As a Jupyter notebook** via [Jupytext](https://github.com/mwouts/jupytext)
- **As a regular Python script**, e.g. `python examples/inconsistencies_example.py`

### [Inconsistencies example](examples/inconsistencies_example.py)

Demonstrates how to input a cell design from literature, inspect flagged inconsistencies from rounded values, resolve them by removing offending inputs, and explore the full set of attributes and equations.

### [Tuning example](examples/tuning_example.py)

Demonstrates how to use the tuning function to solve for a design parameter that satisfies a target constraint, such as finding the cathode thickness needed to achieve a specific cell mass.

### [Sweep example](examples/sweep_example.py)

Demonstrates how to programmatically sweep a design parameter (e.g. cathode film thickness) and observe the resulting changes in cell properties.

## Limitations and Future Work

- **Better cell summary outputs** — Improve the formatting and content of printed summaries.
- **User-defined equations and attributes** — Allow users to add, remove, or modify equations and attributes to create custom cell designs.
- **Hierarchical and modular architecture** — Extend the framework to support more complex designs such as cylindrical cells, cells with tabs, series/parallel configurations, and non-electrochemical parameters like labor and supply chain costs.
- **Multi-unknown solver** — Enable the solver to handle multiple unknowns simultaneously, allowing fully top-down calculations without relying on the tuning function.
- **Better documentation of equations** — Provide clearer explanations of the algebraic relationships and the physical meaning of each attribute.
- **Improved inconsistency and error analysis** — Offer more detailed diagnostics to help users understand the calculation chain and identify sources of error in their inputs.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use this tool in your research, please cite:

> Martin Brischetto, *Battery_Pack_Solver: A universal pouch cell energy density calculator*, 2026. <https://github.com/marbris/Battery_Pack_Solver>
