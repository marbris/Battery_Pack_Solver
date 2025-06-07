# Battery_Pack_Solver/component.py
import numpy as np
from colorama import Fore, Style
from .equation_solver import solve_equation_attr2 as solve_equation_attr
from .component_utils import remove_color_codes
import scipy.optimize


class Component:
    def __init__(self, attributes):
        self._attributes = attributes
        self.reset_attributes()  # Initialize attributes from the provided dictionary
        # self.tolerance = 1e-6  # Allowable relative error when checking consistency
        self.equations = []  # List to hold equations

    def reset_attributes(self):
        # Iterate over all attributes in __dict__ and delete them
        for attr in list(self.__dict__.keys()):
            if attr != "_attributes":  # Don't clear _attributes
                delattr(self, attr)  # Delete the attribute

        # Re-add all attributes from self._attributes
        for key, value in self._attributes.items():
            setattr(self, key, value)

    def solve_equation(self, eq):
        """
        Solve an equation using the external solver function and update the respective lists.
        """

        status, solved_var, num_unknowns = solve_equation_attr(eq)

        return status, solved_var, num_unknowns

    def solve_component(self, verbose=False):
        """
        Iteratively solve all possible values and check consistency.
        If nothing is changed, it returns False
        """
        self.inconsistent = []  # Store any inconsistent equations
        # Track variables that are involved in equations but not solved yet
        self.unsolved_variables = set()
        self.equation_status = []  # Store status of each equation

        colors = {
            "solved": Fore.YELLOW,
            "derived": Fore.BLUE,
            "unsolved": Fore.RED,
            "initial": Fore.GREEN,
            "constant": Fore.WHITE,
        }

        if verbose == "vv":
            print("#" * (len(self.Description) + len("Solving ") + 2 * (1 + 7)))
            print(f"####### Solving {self.Description} #######")
            print("#" * (len(self.Description) + len("Solving ") + 2 * (1 + 7)))

        # Now solve each equation based on its type
        solved = True
        anything_solved = False
        iteration = 0
        while solved:
            solved = False
            iteration += 1
            if verbose == "vv":
                print(f"####### Iteration {iteration} #######")
            # for eq in self.products:
            for eq in self.equations:
                # Solve the equation
                status, solved_var, num_unknowns = self.solve_equation(eq)

                # print(solved_var)
                # adding colors to the terms based on their status:
                cols = []
                for term in eq[1:]:
                    if (status == "solved") and ((term[0], term[1]) == solved_var):
                        cols.append(colors["solved"])
                    else:
                        _, stat, _ = self.get_attribute(term)
                        cols.append(colors[stat])

                _, _, equation_str, _ = self.get_equation(eq, colors=cols)

                # if (self, "w_in_0", "+") in eq:
                # print(f"{eq = }")
                # print(f"{status = }")
                # print(f"{num_unknowns = }")
                # _, stat, _ = self.get_attribute((self, "w_in_0", "+"))
                # print(f"{stat = }")
                if status == "inconsistent":
                    self.inconsistent.append(
                        (eq, (self._attributes[var] for var in eq))
                    )
                    if verbose == "vvv" or verbose == "v":
                        print(f"{equation_str:<60} is inconsistent")

                if status == "consistent":
                    if verbose == "vvv":
                        print(f"{equation_str:<60} is consistent")

                if status == "unsolved":
                    # add the unsolved variables to the set
                    unsolved_vars = (var for var in eq if var not in self._attributes)
                    self.unsolved_variables.update(unsolved_vars)

                    if verbose == "vvv":
                        print(f"{equation_str:<60} unsolved")

                if status == "solved":
                    # Remove solved variables from unsolved set
                    self.unsolved_variables.difference_update(eq)
                    solved = True
                    anything_solved = True

                    if solved_var is not None:
                        value = getattr(solved_var[0], solved_var[1], None)
                        # solved_var = f"{solved_var[0].desc}.{solved_var[1]}"
                        solved_val = f"{value:0.8g}"
                        # solved_str = f"{colors[status]}{solved_var} = {solved_val}{Style.RESET_ALL}"
                        solved_str = f"{colors[status]}{solved_val}{Style.RESET_ALL}"

                    if verbose == "vvv" or verbose == "v":
                        print(f"{equation_str:<70}{solved_str:>25}")

        return anything_solved  # Return True if any variable was solved

    def get_attribute(self, term, color=None):
        # Check if its a constant
        obj, attr, _ = term
        if obj == "const":
            term_str = f"{term[1]}"
            value = attr
            status = "constant"  # Status for constants
        else:
            if attr in obj._attributes:
                status = "initial"
                value = obj._attributes[attr]
            # If it's a derived attribute (property)
            elif hasattr(obj, attr):
                status = "derived"
                value = getattr(obj, attr)  # Call the property method
            else:
                status = "unsolved"
                value = None

            term_str = f"{obj.desc}.{attr}"

        if color:
            term_str = f"{color}{term_str}{Style.RESET_ALL}"

        return value, status, term_str

    # get list of all attributes in the list of equations
    def get_attribute_info(self, ignore_constants=True, color_output=False):
        attribute_info = set()
        for eq in self.equations:
            for term in eq[1:]:
                value, status, term_str = self.get_attribute(term)
                if not (term[0] == "const" and ignore_constants):
                    attribute_info.add(
                        (term[0], term[1], value, status, term_str)
                    )  # Add the description of the object

        return attribute_info

    # print attributes of the component
    # color them based on whether they are solved or not
    def print_attributes(self, getstring=False):
        attributes_info = self.get_attribute_info()
        # Sort by status (initial > derived > unsolved) and then alphabetically by attribute
        status_color = {
            "initial": Fore.GREEN,
            "derived": Fore.BLUE,
            "unsolved": Fore.RED,
        }
        status_order = {"initial": 0, "derived": 1, "unsolved": 2}
        sorted_attributes = sorted(
            attributes_info, key=lambda x: (status_order[x[3]], x[1])
        )

        # Print all attributes with alignment
        max_attr_len = max(len(attr) for _, attr, _, _, _ in sorted_attributes)
        max_value_len = max(len(str(value)) for _, _, value, _, _ in sorted_attributes)

        # Start accumulating the result string
        result = ""
        result += f"{'Attribute':<{max_attr_len}}  {'Value':<{max_value_len}}  Status\n"
        result += "-" * (max_attr_len + max_value_len + 11) + "\n"  # Divider

        # Loop over the sorted attributes and add them to the result string
        for _, _, value, status, term_str in sorted_attributes:

            value_str = f"{value:.5f}" if value is not None else ""
            color = status_color.get(
                status, Fore.RED
            )  # Default to red if status not found

            result += f"{color}{term_str:<{max_attr_len}}  {value_str:<{max_value_len}}  {status.capitalize()}{Style.RESET_ALL}\n"

        # If getstring is True, return the accumulated string; otherwise, print it
        if getstring:
            return result
        else:
            print(result)

    def get_equation(self, eq, colors=False, rtol=1e-5, atol=1e-8):
        """
        Generate an equation string for given terms
        """
        coldict = {
            "derived": Fore.BLUE,
            "unsolved": Fore.RED,
            "initial": Fore.GREEN,
            "constant": Fore.WHITE,
        }
        # operator string
        if eq[0] == "product":
            operator = " * "
        elif eq[0] == "sum":
            operator = " + "

        # Build the equation string
        lhs_termlist = []
        rhs_termlist = []
        # Building the counts
        initial_count = 0
        derived_count = 0
        unsolved_count = 0
        # getting the numbers
        lhs_values = []
        rhs_values = []
        for i, term in enumerate(eq[1:]):
            # getting attribute info
            value, status, term_str = self.get_attribute(term)

            # if there are colors, add them to the term string
            if colors:
                if isinstance(colors, list):
                    term_str = f"{colors[i]}{term_str}{Style.RESET_ALL}"
                else:
                    term_str = f"{coldict[status]}{term_str}{Style.RESET_ALL}"

            # add the term to the correct side of the equation
            if term[2] == "+":
                lhs_termlist.append(term_str)
                lhs_values.append(value)
            else:
                rhs_termlist.append(term_str)
                rhs_values.append(value)

            if status == "initial":
                initial_count += 1
            elif status == "derived":
                derived_count += 1
            elif status == "unsolved":
                unsolved_count += 1

        left_side = operator.join(lhs_termlist)  # lhs terms separated by operator
        right_side = operator.join(rhs_termlist)  # rhs terms separated by operator
        equation_str = f"{left_side} = {right_side}"

        term_status_count = (initial_count, derived_count, unsolved_count)

        if unsolved_count > 0:
            status = "unsolved"
        else:
            if eq[0] == "product":
                if np.isclose(
                    np.prod(lhs_values), np.prod(rhs_values), rtol=rtol, atol=atol
                ):
                    status = "consistent"
                else:
                    status = "inconsistent"
            elif eq[0] == "sum":
                if np.isclose(
                    np.sum(lhs_values), np.sum(rhs_values), rtol=rtol, atol=atol
                ):
                    status = "consistent"
                else:
                    status = "inconsistent"

        values = lhs_values + rhs_values

        return values, status, equation_str, term_status_count

    def print_equations(self, getstring=False):
        """
        Sort equations first by status (consistent/unsolved/inconsistent), then by the
        number of variables that are "initial", "derived", and "unsolved".
        """

        equations_info = [
            (i, eq, *self.get_equation(eq, colors=True))
            for i, eq in enumerate(self.equations)
        ]
        sortkey = {
            eq[0]: (
                # Sorting by status: 'inconsistent' > 'consistent' > 'unsolved'
                {"inconsistent": 0, "consistent": 1, "unsolved": 2}[eq[3]],
                # Sorting by number of initial, derived, and unsolved variables
                *eq[5],
            )
            for eq in equations_info
        }

        # Sort equations based on their status and the count of "initial", "derived", and "unsolved"
        equations_info.sort(key=lambda eq: sortkey[eq[0]])

        # print([len(eq) for eq in equations_info])
        # print([eq[4] for eq in equations_info])
        # the equations have a different length when color codes are removed
        max_eq_len_header = np.max(
            [len(remove_color_codes(eq[4])) for eq in equations_info]
        )
        max_eq_len = np.max([len(eq[4]) for eq in equations_info])
        max_status_len = np.max([len(eq[3]) for eq in equations_info])
        # max_status_len = max(len(status_header), max(len(remove_color_codes(row.split()[-1])) for row in table_body))

        max_eq_len_header += 0
        max_eq_len += 0
        max_status_len += 0

        max_eq_len = 90
        max_status_len = 12

        # Prepare the result string
        result = ""
        result += f"{'Equations':<{max_eq_len_header}} {'Status':<{max_status_len}}\n"
        result += "-" * (max_eq_len_header + max_status_len) + "\n"  # Divider

        for eq in equations_info:
            equation_str = eq[4]
            status = eq[3]

            # Assign colors based on status
            if status == "consistent":
                color = Fore.GREEN
            elif status == "unsolved":
                color = Fore.RED
            elif status == "inconsistent":
                color = Fore.YELLOW
            else:
                color = Fore.WHITE  # Default color for any other status

            # Add the equation and its colored status to the result string
            result += f"{equation_str:<{max_eq_len}}{color}{status.capitalize():<{max_status_len}}{Style.RESET_ALL}\n"
            # result += f'{({"inconsistent": 0, "consistent": 1, "unsolved": 2}[eq["status"]],*count_variable_status(eq["equation"]))}'

        # If getstring is True, return the formatted string; otherwise, print it
        if getstring:
            return result
        else:
            print(result)

    def print_attr_eq(self, getstring=False):
        # Construct the string for the header
        repr_str = "\n"
        repr_str += "#" * (len(self.Description) + 2 * (1 + 7)) + "\n"
        repr_str += f"####### {self.Description} #######\n"
        repr_str += "#" * (len(self.Description) + 2 * (1 + 7)) + "\n"

        # Add the attributes section
        repr_str += "\n####### Attributes #######\n"
        repr_str += self.print_attributes(getstring=True) + "\n"

        # Add the equations section
        repr_str += "####### Equations #######\n"
        repr_str += self.print_equations(getstring=True) + "\n"

        if getstring:
            return repr_str
        else:
            print(repr_str)

    def __repr__(self):
        """
        Return a string representation of the Component object, showing the attributes and equations.
        """
        # Construct the string for the header
        repr_str = self.Description

        # Return the full representation
        return repr_str


class Electrode(Component):
    def __init__(self, attributes):
        super().__init__(attributes)
        self.initialize()  # Initialize attributes and other parameters

    def initialize(self, attributes=None):
        if attributes is None:
            attributes = self._attributes
        super().__init__(attributes)
        self.equations = []  # List to hold product equations

        # if a list of material ratios and densities or costs is provided,
        if "w_i" in self._attributes:

            # if its a value or a single valued list, convert it to a list of two values
            # these correspond to the active and inactive material ratios
            # the assumption is that the first value is the active material ratio
            #
            # if its already a list, then normalize their values to sum to 1
            if isinstance(self.w_i, list):
                if len(self.w_i) == 1:  # Single valued list
                    self.w_i = [
                        self.w_i[0],
                        1 - self.w_i[0],
                    ]
                elif len(self.w_i) > 1:  # Multiple values in the list
                    total = sum(self.w_i)
                    self.w_i = [value / total for value in self.w_i]
            elif isinstance(self.w_i, (int, float)):
                # If it's a number, convert it to [value, 1-value]
                self.w_i = [
                    self.w_i,
                    1 - self.w_i,
                ]

            self.N_comp = len(self.w_i)
            N_comp = self.N_comp  # Number of components

            # I create a new attribute for each component in the list
            # if its only two, its am and in (inacitve)
            # if its more than two, its am, in_0, in_1, ..., in_(N_comp-1)
            # they all have to add up to one.
            w_sum_terms = []
            for i in range(N_comp):
                if i == 0:
                    label = "am"
                elif N_comp == 2:
                    label = "in"
                else:
                    label = f"in_{i-1}"
                self._attributes[f"w_{label}"] = self.w_i[i]
                setattr(self, f"w_{label}", self.w_i[i])
                w_sum_terms.append((self, f"w_{label}", "+"))
                self.equations += [
                    # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                    (
                        "product",
                        (self, "M", "+"),
                        (self, f"w_{label}", "+"),
                        (self, f"M_{label}", "-"),
                    ),
                    (
                        "product",
                        (self, "Mf", "+"),
                        (self, f"w_{label}", "+"),
                        (self, f"Mf_{label}", "-"),
                    ),
                ]

            # adding the equation for the sum of all component ratios
            # sum wi = 1
            self.equations.append(("sum", *w_sum_terms, ("const", 1, "-")))
            # if there are more than two components,
            # w_in is the sum of the inactive materials:
            if N_comp > 2:
                self.equations.append(("sum", *w_sum_terms[1:], (self, "w_in", "-")))

            # if the active material ratio is one,
            # im setting the inactive volume, mass and dollar value to zero
            if self.w_i[0] == 1:
                self.V_in = 0
                self.w_in = 0
                self.D_in = 0
                self.M_in = 0
            # calculate the solid phase density
            if "Mv_i" in self._attributes:
                # if its not a list, i make it into a list
                if isinstance(self.Mv_i, (int, float)):
                    self.Mv_i = [self.Mv_i]

                v_sum_terms = []
                for i in range(N_comp):
                    if i == 0:
                        label = "am"
                    elif N_comp == 2:
                        label = "in"
                    else:
                        label = f"in_{i-1}"

                    # I only add the value if it exists among the components
                    # otherwise it will be skipped, and remains unknown
                    if i < len(self.Mv_i):
                        self._attributes[f"Mv_{label}"] = self.Mv_i[i]
                        setattr(self, f"Mv_{label}", self.Mv_i[i])

                    # the equation is:
                    # I need the Volume of component i per electrode mass
                    # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                    self.equations += [
                        # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                        (
                            "product",
                            (self, f"Mv_{label}", "+"),
                            (self, f"Vm_{label}", "+"),
                            (self, f"w_{label}", "-"),
                        ),
                    ]
                    # self.products.append(
                    # (
                    # (self, f"Mv_{label}"),
                    # (self, f"Vm_{label}"),
                    # (self, f"w_{label}"),
                    # )
                    # )
                    v_sum_terms.append((self, f"Vm_{label}", "+"))

                # sum (Vol i/Mass c) = (Vol solid phase/Mass c)
                self.equations.append(("sum", *v_sum_terms, (self, "Vm_solid", "-")))
                # (Mass c/ Vol solid phase) * (Vol solid phase/Mass c) = 1
                self.equations.append(
                    (
                        "product",
                        (self, "Mv_solid", "+"),
                        (self, "Vm_solid", "+"),
                        ("const", 1, "-"),
                    )
                )

            # calculate material costs
            if "Dm_i" in self._attributes:
                # if its not a list, i make it into a list
                if isinstance(self.Dm_i, (int, float)):
                    self.Dm_i = [self.Dm_i]
                d_sum_terms = []
                for i in range(N_comp):
                    if i == 0:
                        label = "am"
                    elif N_comp == 2:
                        label = "in"
                    else:
                        label = f"in_{i-1}"
                    if i < len(self.Dm_i):
                        self._attributes[f"Dm_{label}"] = self.Dm_i[i]
                        setattr(self, f"Dm_{label}", self.Dm_i[i])

                    # (Cost i/Mass i)*(Mass i/Mass c)*(Mass c/film) = (Cost i/film)
                    self.equations += [
                        (
                            "product",
                            (self, f"Dm_{label}", "+"),
                            (self, f"w_{label}", "+"),
                            (self, "Mf", "+"),
                            (self, f"Df_{label}", "-"),
                        ),
                        (
                            "product",
                            (self, f"Df_{label}", "+"),
                            (self, "N", "+"),
                            (self, f"D_{label}", "-"),
                        ),
                    ]
                    d_sum_terms.append((self, f"Df_{label}", "+"))

                # sum of all component costs per film = total cost of film
                self.equations.append(("sum", *d_sum_terms, (self, "Df", "-")))

        self.equations += [
            # Area of Film
            ("product", (self, "d0", "+"), (self, "d1", "+"), (self, "A", "-")),
            # Volume of Film
            ("product", (self, "Lf", "+"), (self, "A", "+"), (self, "Vf", "-")),
            # Pore Volume of Film
            ("product", (self, "Vf", "+"), (self, "por", "+"), (self, "Vf_por", "-")),
            # Solid Phase Volume of Film
            (
                "product",
                (self, "Vf", "+"),
                (self, "por_inv", "+"),
                (self, "Vf_solid", "-"),
            ),
            # (1-por) * Solid Phase Density = film density
            (
                "product",
                (self, "por_inv", "+"),
                (self, "Mv_solid", "+"),
                (self, "Mv", "-"),
            ),
            # film Volume * film Density =  film Mass
            ("product", (self, "Vf", "+"), (self, "Mv", "+"), (self, "Mf", "-")),
            # Mass of Film * Active Material Ratio = Active Material Mass
            ("product", (self, "Mf", "+"), (self, "w_am", "+"), (self, "Mf_am", "-")),
            # Mass of Film * Inactive Material Ratio = Inactive Material Mass
            ("product", (self, "Mf", "+"), (self, "w_in", "+"), (self, "Mf_in", "-")),
            # Areal Mass of Film * Area = Total Mass of Electrode
            ("product", (self, "Ma", "+"), (self, "A", "+"), (self, "Mf", "-")),
            # Film Volume * Film Density = Film Mass
            ("product", (self, "Vf", "+"), (self, "Mv", "+"), (self, "Mf", "-")),
            # Active Material Specific Capacity * Active Material Mass = Film Capacity
            ("product", (self, "Qm_am", "+"), (self, "Mf_am", "+"), (self, "Qf", "-")),
            # Areal Capacity * Area = Film Capacity
            ("product", (self, "Qa", "+"), (self, "A", "+"), (self, "Qf", "-")),
            # Average Voltage * capacity = Specific Energy
            (
                "product",
                (self, "Vavg", "+"),
                (self, "Qm_am", "+"),
                (self, "Em_am", "-"),
            ),
            # Active Material Mass * Active Material Specific Capacity = Film Capacity
            ("product", (self, "Qm_am", "+"), (self, "Mf_am", "+"), (self, "Qf", "-")),
            # Active Material Mass * Active Material Specific Capacity = Film Capacity
            ("product", (self, "Em_am", "+"), (self, "Mf_am", "+"), (self, "Ef", "-")),
            # Mass of Film * Number of Films = Total Mass of Electrode
            ("product", (self, "Mf", "+"), (self, "N", "+"), (self, "M", "-")),
            # Mass of Cathode * Active Mass ratio = Active Material Mass of Electrode
            ("product", (self, "M", "+"), (self, "w_am", "+"), (self, "M_am", "-")),
            # Areal Mass of Electrode * Active Mass ratio = Areal Active Mass Material Mass of Electrode
            ("product", (self, "Ma", "+"), (self, "w_am", "+"), (self, "Ma_am", "-")),
            # Mass of Cathode * Inactive Mass ratio = Inactive Material Mass of Electrode
            ("product", (self, "M", "+"), (self, "w_in", "+"), (self, "M_in", "-")),
            # Active Material Mass of Film * Number of Films = Total Active Material Mass of Electrode
            ("product", (self, "Mf", "+"), (self, "N", "+"), (self, "M", "-")),
            # Active Material Mass of Film * Number of Films = Total Active Material Mass of Electrode
            ("product", (self, "Mf_am", "+"), (self, "N", "+"), (self, "M_am", "-")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            ("product", (self, "Mf_in", "+"), (self, "N", "+"), (self, "M_in", "-")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            ("product", (self, "Mf_in", "+"), (self, "N", "+"), (self, "M_in", "-")),
            # Film Thickness * Number of Films = Total Thickness of Electrode
            ("product", (self, "Lf", "+"), (self, "N", "+"), (self, "L", "-")),
            # Film Volume * Number of Films = Total Volume of Electrode
            ("product", (self, "Vf", "+"), (self, "N", "+"), (self, "V", "-")),
            # Film Pore Volume * Number of Films = Total Pore Volume of Electrode
            ("product", (self, "Vf_por", "+"), (self, "N", "+"), (self, "V_por", "-")),
            # Film Capacity * Number of Films = Total Capacity of Electrode
            ("product", (self, "Qf", "+"), (self, "N", "+"), (self, "Q", "-")),
            # Film Energy * Number of Films = Total Energy of Electrode
            ("product", (self, "Ef", "+"), (self, "N", "+"), (self, "E", "-")),
            # Active Material Ratio AM_ratio = w_am/w_in
            # ((self, "AM_ratio"), (self, "w_in"), (self, "w_am")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ("product", (self, "Dm_am", "+"), (self, "M_am", "+"), (self, "D_am", "-")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            (
                "product",
                (self, "Dm_am", "+"),
                (self, "Mf_am", "+"),
                (self, "Df_am", "-"),
            ),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ("product", (self, "Dm", "+"), (self, "Mf", "+"), (self, "Df", "-")),
            # Film Cost * Number of Films = Electrode Cost
            ("product", (self, "Df", "+"), (self, "N", "+"), (self, "D", "-")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            # (Vol in) * (Mass in / Vol in) = Mass in
            ("product", (self, "V_in", "+"), (self, "Mv_in", "+"), (self, "M_in", "-")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ("product", (self, "V_am", "+"), (self, "Mv_am", "+"), (self, "M_am", "-")),
            # porosity + inverse porosity = 1, useful for density calculations
            ("sum", (self, "por", "+"), (self, "por_inv", "+"), ("const", 1, "-")),
            # active material cost + inactive material cost = total material cost
            ("sum", (self, "Df_am", "+"), (self, "Df_in", "+"), (self, "Df", "-")),
            # active material cost + inactive material cost = total material cost
            ("sum", (self, "D_am", "+"), (self, "D_in", "+"), (self, "D", "-")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ("sum", (self, "V_am", "+"), (self, "V_in", "+"), (self, "V", "-")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ("sum", (self, "M_am", "+"), (self, "M_in", "+"), (self, "M", "-")),
        ]

    def clear_and_reinitialize(self):
        """
        Clears all instance attributes except _attributes, and then reinitializes them.
        """
        # Clear all attributes except _attributes
        self.reset_attributes()
        # Reinitialize the object
        self.initialize()


class Inactive(Component):
    def __init__(self, attributes):
        super().__init__(attributes)
        self.initialize()  # Initialize attributes and other parameters

    def initialize(self, attributes=None):
        if attributes is None:
            attributes = self._attributes
        super().__init__(attributes)

        self.equations = [
            # side1 * side2 = Area
            ("product", (self, "d0", "+"), (self, "d1", "+"), (self, "A", "-")),
            # Film Thickness * Area = Film Volume
            ("product", (self, "Lf", "+"), (self, "A", "+"), (self, "Vf", "-")),
            # Areal Mass of Film * Area = Total Mass of Film
            ("product", (self, "Ma", "+"), (self, "A", "+"), (self, "Mf", "-")),
            # Film Volume * Film Density = Film Mass
            ("product", (self, "Vf", "+"), (self, "Mv", "+"), (self, "Mf", "-")),
            # Mass of Film * Number of Films = Total Mass of Electrode
            ("product", (self, "Mf", "+"), (self, "N", "+"), (self, "M", "-")),
            # Film Thickness * Number of Films = Total Thickness of Electrode
            ("product", (self, "Lf", "+"), (self, "N", "+"), (self, "L", "-")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ("product", (self, "Dm", "+"), (self, "Mf", "+"), (self, "Df", "-")),
            # Film Cost * Number of Films = Electrode Cost
            ("product", (self, "Df", "+"), (self, "N", "+"), (self, "D", "-")),
            # Film Volume * Porosity = Film Pore Volume
            ("product", (self, "Vf", "+"), (self, "por", "+"), (self, "Vf_por", "-")),
            # Solid Phase Volume of Film
            (
                "product",
                (self, "Vf", "+"),
                (self, "por_inv", "+"),
                (self, "Vf_solid", "-"),
            ),
            # (1-por) * Solid Phase Density = film density
            (
                "product",
                (self, "por_inv", "+"),
                (self, "Mv_solid", "+"),
                (self, "Mv", "-"),
            ),
            # Film Pore Volume * Number of Films = Total Pore Volume of Electrode
            ("product", (self, "Vf_por", "+"), (self, "N", "+"), (self, "V_por", "-")),
            # Inverse porosity, useful for density calculations
            ("sum", (self, "por", "+"), (self, "por_inv", "+"), ("const", 1, "-")),
        ]

    def clear_and_reinitialize(self):
        """
        Clears all instance attributes except _attributes, and then reinitializes them.
        """
        # Clear all attributes except _attributes
        self.reset_attributes()
        # Reinitialize the object
        self.initialize()


class Electrolyte(Component):
    def __init__(self, attributes):
        super().__init__(attributes)
        self.initialize()  # Initialize attributes and other parameters

    def initialize(self, attributes=None):
        if attributes is None:
            attributes = self._attributes
        super().__init__(attributes)
        # equations A * B = C
        self.equations = [
            # Volume, Density, Mass relationship
            ("product", (self, "V", "+"), (self, "Mv", "+"), (self, "M", "-")),
            # Cost/Mass, Mass, Cost
            ("product", (self, "Dm", "+"), (self, "M", "+"), (self, "D", "-")),
        ]

    def clear_and_reinitialize(self):
        """
        Clears all instance attributes except _attributes, and then reinitializes them.
        """
        # Clear all attributes except _attributes
        self.reset_attributes()
        # Reinitialize the object
        self.initialize()


class PouchCell(Component):
    def __init__(self, attributes, verbose=False):
        super().__init__(attributes)
        self.initialize_components()  # Initialize components
        self.initialize_attributes()  # Initialize attributes

        self.solve_all(verbose=verbose)  # Solve and validate all components

        if "tune" in self._attributes:

            # if its not a list, make it a list with one item
            if not isinstance(self._attributes["tune"], list):
                self._attributes["tune"] = [self._attributes["tune"]]

            # optimize each item in the list
            for opt in self._attributes["tune"]:
                tune_obj_str, tune_attr = opt.get("tuning_variable")
                opt_str, opt_attr = opt.get("optimization_variable")
                guess = opt.get("guess")
                target = opt.get("target")

                # the first element of the tuple is the object, the second is the attribute
                tune_obj = self.components_dict.get(tune_obj_str)
                opt_obj = self.components_dict.get(opt_str)
                self.optimize(
                    (tune_obj, tune_attr),  # Tuning variable
                    (opt_obj, opt_attr),  # Optimization variable
                    guess,  # Initial guess for tuning variable
                    target,  # Target value for optimization variable
                )

                # at the end of each optimization step,
                # I add the tuned value to the _attributes dict
                # so that it survives the next step
                #   (each optimize() wipes everything clean
                #   and reinitializes from the _attributes dict)
                # the components are stored in sub-dicts of _attributes
                if tune_obj_str != "P":
                    self._attributes["components"][tune_obj_str][tune_attr] = getattr(
                        tune_obj, tune_attr
                    )
                else:
                    self._attributes[tune_attr] = getattr(tune_obj, tune_attr)

    def initialize_components(self):
        self.c = Electrode(self._attributes["components"]["c"])
        self.a = Electrode(self._attributes["components"]["a"])
        self.el = Electrolyte(self._attributes["components"]["el"])

        self.a_cc = Inactive(self._attributes["components"]["c_cc"])
        self.c_cc = Inactive(self._attributes["components"]["a_cc"])
        self.s = Inactive(self._attributes["components"]["s"])
        self.pg = Inactive(self._attributes["components"]["pg"])

        self.components_dict = {
            "c": self.c,
            "a": self.a,
            "el": self.el,
            "pg": self.pg,
            "s": self.s,
            "a_cc": self.a_cc,
            "c_cc": self.c_cc,
            "P": self,
        }

    def initialize_attributes(self):
        ### N is defined as the number of double sided anodes ###
        self.a_cc._attributes["N"] = self.N
        self.a._attributes["N"] = 2 * self.N
        # One cathode and separator for each anode
        self.c._attributes["N"] = 2 * self.N
        # Separator is between each anode and cathode.N
        self.s._attributes["N"] = 2 * self.N
        # One more cathode foil, because its at the end of the cell
        self.c_cc._attributes["N"] = self.N + 1
        # there are two paackaging layers per pouch cell
        self.pg._attributes["N"] = 2

        # I need to set these as paramteres as well.
        # this is a workaround until if fix a bug.
        # _attributes[] are only set as attributes during initialization
        # so i need to set them here as well
        ### N is defined as the number of double sided anodes ###
        self.a_cc.N = self.N
        self.a.N = 2 * self.N
        # One cathode and separator for each anode
        self.c.N = 2 * self.N
        # Separator is between each anode and cathode.N
        self.s.N = 2 * self.N
        # One more cathode foil, because its at the end of the cell
        self.c_cc.N = self.N + 1
        # there are two paackaging layers per pouch cell
        self.pg.N = 2

        self.components_dict = {
            "c": self.c,
            "a": self.a,
            "el": self.el,
            "pg": self.pg,
            "s": self.s,
            "a_cc": self.a_cc,
            "c_cc": self.c_cc,
            "P": self,
        }

        if "d0_c" in self._attributes and "d1_c" in self._attributes:
            self.c._attributes["d0"] = self._attributes["d0_c"]
            self.c._attributes["d1"] = self._attributes["d1_c"]
            self.c_cc._attributes["d0"] = self._attributes["d0_c"]
            self.c_cc._attributes["d1"] = self._attributes["d1_c"]
            setattr(self.c, "d0", self._attributes["d0_c"])
            setattr(self.c, "d1", self._attributes["d1_c"])
            setattr(self.c_cc, "d0", self._attributes["d0_c"])
            setattr(self.c_cc, "d1", self._attributes["d1_c"])

        if "d0_a" in self._attributes and "d1_a" in self._attributes:
            self.a._attributes["d0"] = self._attributes["d0_a"]
            self.a._attributes["d1"] = self._attributes["d1_a"]
            self.a_cc._attributes["d0"] = self._attributes["d0_a"]
            self.a_cc._attributes["d1"] = self._attributes["d1_a"]
            setattr(self.a, "d0", self._attributes["d0_a"])
            setattr(self.a, "d1", self._attributes["d1_a"])
            setattr(self.a_cc, "d0", self._attributes["d0_a"])
            setattr(self.a_cc, "d1", self._attributes["d1_a"])

        # get equations from components
        self.equations = []
        for comp in self.components_dict.values():
            if comp is not self:
                self.equations += comp.equations

        # add Pouch cell equations
        self.equations += [
            # anode equations
            # Capacity ratio NP of anode and cathode
            # mAh_a/mAh_c, ratio between the capacities of cathode and anode
            # mAh_c/film , specific capacity of cathode film
            # mAh_a/film , specific capacity of anode film
            ("product", (self, "NP", "+"), (self.c, "Qf", "+"), (self.a, "Qf", "-")),
            # mAh_a/g_aam, specific capacity of anode active material
            # (mAh_c/g_cam)/(mAh_a/g_aam), ratio between the specific capacities of cathode and anode active materials
            # mAh_c/g_cam, specific capacity of cathode active material
            (
                "product",
                (self.a, "Qm_am", "+"),
                (self, "Qm_am_ratio", "+"),
                (self.c, "Qm_am", "-"),
            ),
            # mAh_a/mAh_c, ratio between the capacities of cathode and anode
            # (mAh_c/g_cam)/(mAh_a/g_aam), ratio between the specific capacities of cathode and anode active materials
            # g_aam/g_cam, amount of anode active material per cathode active material
            (
                "product",
                (self, "NP", "+"),
                (self, "Qm_am_ratio", "+"),
                (self, "NP_cam", "-"),
            ),
            # g_aam/g_cam, amount of anode active material per cathode active material
            # g_cam, mass active cathode material
            # g_aam/g_cam, amount of anode active material per cathode active material
            (
                "product",
                (self, "NP_cam", "+"),
                (self.c, "M_am", "+"),
                (self.a, "M_am", "-"),
            ),
            # Electrolyte equations
            # Amount of electrolyte in cell by EC and capacity
            ("product", (self, "EC", "+"), (self.c, "Q", "+"), (self.el, "M", "-")),
            # mAh_c/g_cam, specific capacity of cathode active material
            # g_el/mAh_c, amount of electrolyte in cell per capacity of active cathode
            # g_el/g_cam, amount of electrolyte in cell per mass of active cathode material
            (
                "product",
                (self.c, "Qm_am", "+"),
                (self, "EC", "+"),
                (self, "EC_cam", "-"),
            ),
            # g_el/g_cam, amount of electrolyte in cell per mass of active cathode material
            # g_cam, mass active cathode material
            # g_el, mass of electrolyte in cell
            (
                "product",
                (self, "EC_cam", "+"),
                (self.c, "M_am", "+"),
                (self.el, "M", "-"),
            ),
            # g_am/g_cam, Total Active Mass of Cell per active cathode material mass
            # g_cam, mass active cathode material
            # g_am, mass of active material in cell
            (
                "product",
                (self, "M_active_cam", "+"),
                (self.c, "M_am", "+"),
                (self, "M_active", "-"),
            ),
            # energy density of cell
            ("product", (self, "Em", "+"), (self, "M", "+"), (self, "E", "-")),
            # Capacity of cell
            ("product", (self, "Qm", "+"), (self, "M", "+"), (self, "Q", "-")),
            # Area * Thickness = Volume
            ("product", (self, "A", "+"), (self, "L", "+"), (self, "V", "-")),
            # Active Material material ratio of cell
            (
                "product",
                (self, "M", "+"),
                (self, "M_am_ratio", "+"),
                (self, "M_active", "-"),
            ),
            # Total Energy in pouch cell
            ("product", (self, "Q", "+"), (self, "Vavg", "+"), (self, "E", "-")),
            # Total energy per volume in pouch cell
            ("product", (self, "Ev", "+"), (self, "V", "+"), (self, "E", "-")),
            # energy density in the limit of no inactive materials
            (
                "product",
                (self, "Em_active", "+"),
                (self, "M_active", "+"),
                (self, "E", "-"),
            ),
            # Total Mass of Cell
            (
                "sum",
                (self.el, "M", "+"),
                (self.c, "M", "+"),
                (self.a, "M", "+"),
                (self.a_cc, "M", "+"),
                (self.c_cc, "M", "+"),
                (self.s, "M", "+"),
                (self.pg, "M", "+"),
                (self, "M", "-"),
            ),
            # Total Active Mass of Cell
            (
                "sum",
                (self.el, "M", "+"),
                (self.c, "M_am", "+"),
                (self.a, "M_am", "+"),
                (self, "M_active", "-"),
            ),
            # Total Inactive Mass of Cell
            (
                "sum",
                (self.c, "M_in", "+"),
                (self.a, "M_in", "+"),
                (self.a_cc, "M", "+"),
                (self.c_cc, "M", "+"),
                (self.s, "M", "+"),
                (self.pg, "M", "+"),
                (self, "M_inactive", "-"),
            ),
            # Total Mass of Cell
            (
                "sum",
                (self, "M_inactive", "+"),
                (self, "M_active", "+"),
                (self, "M", "-"),
            ),
            # Total Inactive Mass in Electrodes
            (
                "sum",
                (self.c, "M_in", "+"),
                (self.a, "M_in", "+"),
                (self, "M_ac_inactive", "-"),
            ),
            # Total Inactive Mass in Cell excluding Electrodes
            (
                "sum",
                (self.a_cc, "M", "+"),
                (self.c_cc, "M", "+"),
                (self.s, "M", "+"),
                (self.pg, "M", "+"),
                (self, "M_not_ac_inactive", "-"),
            ),
            # Total Inactive Mass of Cell
            (
                "sum",
                (self, "M_ac_inactive", "+"),
                (self, "M_not_ac_inactive", "+"),
                (self, "M_inactive", "-"),
            ),
            # Total Cost of Cell
            (
                "sum",
                (self.el, "D", "+"),
                (self.c, "D", "+"),
                (self.a, "D", "+"),
                (self.a_cc, "D", "+"),
                (self.c_cc, "D", "+"),
                (self.s, "D", "+"),
                (self.pg, "D", "+"),
                (self, "D", "-"),
            ),
            # Cost of active materials in Cell
            (
                "sum",
                (self.el, "D", "+"),
                (self.c, "D_am", "+"),
                (self.a, "D_am", "+"),
                (self, "D_active", "-"),
            ),
            # Total Inactive Mass of Cell
            (
                "sum",
                (self.c, "D_in", "+"),
                (self.a, "D_in", "+"),
                (self.a_cc, "D", "+"),
                (self.c_cc, "D", "+"),
                (self.s, "D", "+"),
                (self.pg, "D", "+"),
                (self, "D_inactive", "-"),
            ),
            # Total Cost of Cell
            (
                "sum",
                (self, "D_inactive", "+"),
                (self, "D_active", "+"),
                (self, "D", "-"),
            ),
            # 1 + NP_cam + EC_cam = M_active_cam Total Active Mass of Cell per amount of cathode active material
            (
                "sum",
                ("const", 1, "+"),
                (self, "NP_cam", "+"),
                (self, "EC_cam", "+"),
                (self, "M_active_cam", "-"),
            ),
            # Total inactive thickness of cell
            (
                "sum",
                (self.c_cc, "L", "+"),
                (self.a_cc, "L", "+"),
                (self.s, "L", "+"),
                (self.pg, "L", "+"),
                (self, "L_inactive", "-"),
            ),
            # Total active thickness of cell
            ("sum", (self.c, "L", "+"), (self.a, "L", "+"), (self, "L_active", "-")),
            # Total Pouch cell thickness
            (
                "sum",
                (self, "L_active", "+"),
                (self, "L_inactive", "+"),
                (self, "L", "-"),
            ),
        ]

    def clear_and_reinitialize_pouch_cell(self):
        """
        Clears all instance attributes except _attributes, and then reinitializes them.
        """
        # Clear all attributes except _attributes
        for component in [
            self.c,
            self.a,
            self.el,
            self.a_cc,
            self.c_cc,
            self.s,
            self.pg,
        ]:
            component.clear_and_reinitialize()

        # self.reset_attributes()
        # I need to reset the attributes of the pouch cell itself,
        #   avoiding both _attributes and the components
        # Iterate over all attributes in __dict__ and delete them
        for attr in list(self.__dict__.keys()):
            if attr not in [
                "c",
                "a",
                "el",
                "a_cc",
                "c_cc",
                "s",
                "pg",
                "_attributes",
            ]:  # Don't clear _attributes
                delattr(self, attr)  # Delete the attribute

        # Re-add all attributes from self._attributes
        for key, value in self._attributes.items():
            setattr(self, key, value)
        # Reinitialize the object
        self.initialize_attributes()

    ##########################################
    ######### OTHER PROPERTIES ###############
    ##########################################

    @property  # Area of Pouch Cell is the maximum of cathode and anode area
    def A(self):
        # Check if either self.c.A or self.a.A is None
        if self.c.A is None or self.a.A is None:
            return None
        return np.max([self.c.A, self.a.A])

    @property  # Capacity of the pouch cell is the smallest of cathode and anode capacity
    def Q(self):
        # Check if either self.c.A or self.a.A is None
        if self.c.Q is None or self.a.Q is None:
            return None
        return np.min([self.c.Q, self.a.Q])

    @property  # Energy of the pouch cell is that of the cathode
    def E(self):
        return getattr(self.c, "E", None)

    ##########################################
    ######### OPTIMIZATION ###################
    ##########################################

    def solve_all(self, verbose=False):

        solved = True
        while solved:
            solved = False

            # Solve for its components (Electrode, Inactive, Electrolyte, etc.)
            # solved |= self.a.solve_component(verbose=verbose)
            # solved |= self.c.solve_component(verbose=verbose)
            # solved |= self.el.solve_component(verbose=verbose)

            # solved |= self.a_cc.solve_component(verbose=verbose)
            # solved |= self.c_cc.solve_component(verbose=verbose)
            # solved |= self.s.solve_component(verbose=verbose)
            # solved |= self.pg.solve_component(verbose=verbose)

            # Solve for PouchCell itself
            solved = self.solve_component(verbose=verbose)

    def clear_reinitialize_and_solve_pouch_cell(self, verbose=False):
        # Reinitialize the puch cell so that it cal calculate the new values
        self.clear_and_reinitialize_pouch_cell()
        self.solve_all(verbose=verbose)

    def _solve_and_error(
        self,
        tuning_variable,
        optimization_variable,
        tuning_value,
        optimization_target,
        verbose=False,
    ):
        """
        Function that will set the tuning variable, call solve_all(), and return the error.
        """

        # Reinitialize the puch cell so that it cal calculate the new values
        self.clear_and_reinitialize_pouch_cell()

        if isinstance(tuning_value, np.ndarray):
            # If a list is provided, use the first element as the value
            tuning_value = tuning_value[0]

        # Set the tuning variable (e.g., (self.c, 'Lf') = tuning_variable)
        setattr(tuning_variable[0], tuning_variable[1], tuning_value)

        # Run the solve_all method to update all variables
        self.solve_all(verbose=False)

        # Get the current value of the optimization variable
        current_value = getattr(optimization_variable[0], optimization_variable[1])

        # Compute the squared error
        error = (current_value - optimization_target) ** 2

        if verbose:
            print(
                f"Tuning variable {tuning_variable[0].desc}.{tuning_variable[1]}, to achieve optimization target {optimization_variable[0].desc}.{optimization_variable[1]} = {optimization_target}\n"
                f"Current Value {tuning_variable[0].desc}.{tuning_variable[1]} = {tuning_value}\n"
                f"Target = {optimization_variable[0].desc}.{optimization_variable[1]} = {current_value}\n"
                f"Error = {error}\n"
            )

        return error

    def optimize(
        self,
        tuning_variable,
        optimization_variable,
        initial_guess,
        optimization_target,
        verbose=False,
    ):
        """
        Perform the optimization process using scipy.optimize.minimize to minimize the squared error.
        """
        # Define the lambda function to be minimized by scipy.optimize.minimize
        result = scipy.optimize.minimize(
            lambda x: self._solve_and_error(
                tuning_variable,
                optimization_variable,
                x,
                optimization_target,
                verbose=verbose,
            ),
            initial_guess,
        )

        # After optimization, set the tuning variable to the optimized value
        optimized_value = result.x[0]
        setattr(tuning_variable[0], tuning_variable[1], optimized_value)

        if verbose:
            print(f"Optimization result: {result}")
            print(f"Optimized {tuning_variable[1]} to {optimized_value}")

        # Return the optimized value
        return optimized_value

    ##########################################
    ####### PRINT PROPS ######################
    ##########################################
    def pretty_print_dict(self, d):
        for section, values in d.items():
            print(f"{section.upper()}:")  # Print section header
            print("-" * 50)

            # Determine the maximum label width for alignment
            label_width = max(len(label) for label in values.keys())

            # Iterate through the key-value pairs
            for label, value in values.items():
                # Format the value with 2 decimal places and align to the right
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:>{50-label_width-2}.2f}"  # Align numbers to the right with 2 decimal places
                else:
                    formatted_value = (
                        f"{value:>12}"  # Just align strings to the right as well
                    )

                # Print the label (left-aligned) and the formatted value (right-aligned)
                print(f"{label:<{label_width}}  {formatted_value}")

            print("\n")  # Space between sections

    def print(self, set="libatt"):
        if set == "spreadsheet":
            out = {
                "Cell": {
                    "Mass energy density (Wh/kg)": getattr(self, "Em", np.nan),
                    "Total cell weight (g)": getattr(self, "M", np.nan),
                    "Average Voltage (V)": getattr(self, "Vavg", np.nan),
                    "Capacity (Ah)": getattr(self, "Q", np.nan) * 1e-3,
                    "Thickness (cm)": getattr(self, "L", np.nan),
                },
                "Cathode": {
                    "Discharge Capacity (mAh/g)": getattr(self.c, "Qm_am", np.nan),
                    "AM Loading (mg/cm2)": getattr(self.c, "Ma_am", np.nan) * 1e3,
                    "AM in Cathode (%)": getattr(self.c, "w_am", np.nan) * 100,
                    "Coating Weight each side (mg)": getattr(self.c, "Mf", np.nan)
                    * 1e3,
                    "Coating Capacity each side (mAh)": getattr(self.c, "Qf", np.nan),
                    "Cathode Porosity (%)": getattr(self.c, "por", np.nan) * 100,
                    "Cathode Area (cm2)": getattr(self.c, "A", np.nan),
                    "Number of Single Sided Cathode Layers": 2,
                    "Number of Double Sided Cathode Layers": getattr(
                        self.c, "N", np.nan
                    )
                    - 2,
                },
                "Al-Foil": {
                    "Thickness (μm)": getattr(self.c_cc, "Lf", np.nan) * 1e4,
                    "Areal mass (mg/cm2)": getattr(self.c_cc, "Ma", np.nan) * 1e3,
                },
                "Anode": {
                    "Specific Capacity (mAh/g)": getattr(self.a, "Qm_am", np.nan),
                    "Anode Thickness (μm)": getattr(self.a, "Lf", np.nan) * 1e4,
                    "Anode Capacity each side (mAh)": getattr(self.a, "Qf", np.nan),
                    "N/P ratio": getattr(self, "NP", np.nan),
                    "Number of Double Sided Anode Layers": getattr(self.a, "N", np.nan),
                },
                "Cu": {
                    "Cu foil thickness (μm)": getattr(self.a_cc, "Lf", np.nan) * 1e4,
                    "Cu foil areal mass (mg/cm2)": getattr(self.a_cc, "Ma", np.nan)
                    * 1e3,
                },
                "Electrolyte": {
                    "E/C ratio (g/Ah)": getattr(self, "EC", np.nan) * 1e3,
                    "Mass (g)": getattr(self.el, "M", np.nan),
                },
                "Separator": {
                    "Thickness (μm)": getattr(self.s, "Lf", np.nan) * 1e4,
                },
                "Inactive Material": {
                    "Inactive Mass(g)": getattr(self.pg, "M", np.nan),
                },
                "Details": {
                    "Density of Li (g/cm3)": getattr(self.a, "Mv_solid", np.nan),
                    "Al current collector mass (mg)": getattr(self.c_cc, "Mf", np.nan)
                    * 1e3,
                    "Anode Area (cm2)": getattr(self.a, "A", np.nan),
                    "Cu current collector mass (mg)": getattr(self.a_cc, "Mf", np.nan)
                    * 1e3,
                },
                "Weights": {
                    "Double Sided Cathode (mg)": (
                        getattr(self.c_cc, "Mf", np.nan)
                        + 2 * getattr(self.c, "Mf", np.nan)
                    )
                    * 1e3,
                    "Single Sided Cathode (mg)": (
                        getattr(self.c_cc, "Mf", np.nan) + getattr(self.c, "Mf", np.nan)
                    )
                    * 1e3,
                    "Cathode sheets (mg)": (
                        getattr(self.c, "M", np.nan) + getattr(self.c_cc, "M", np.nan)
                    )
                    * 1e3,
                    "Double Sided Anode (mg)": (
                        getattr(self.a_cc, "Mf", np.nan)
                        + 2 * getattr(self.a, "Mf", np.nan)
                    )
                    * 1e3,
                    "Total anode weight (mg)": (
                        getattr(self.a, "M", np.nan) + getattr(self.a_cc, "M", np.nan)
                    )
                    * 1e3,
                },
            }
            # pprint(out, width=40, sort_dicts=False, indent=8)
            self.pretty_print_dict(out)
        elif set == "libatt":
            out = {
                "Cell": {
                    "Energy density (Wh/kg)": getattr(self, "Em", np.nan),
                    "Volumetric Energy density (Wh/L)": getattr(self, "Ev", np.nan),
                    "Capacity (Ah)": getattr(self, "Q", np.nan) * 1e-3,
                    "Average Voltage (V)": getattr(self, "Vavg", np.nan),
                },
                "Anode design": {
                    "Li thickness (um)": getattr(self.a, "Lf", np.nan) * 1e4,
                    "Specific capacity (mAh/g)": getattr(self.a, "Qm_am", np.nan),
                    "CB (N/P)": getattr(self, "NP", np.nan),
                    "Electrode width (mm)": getattr(self.a, "d0", np.nan) * 10,
                    "Electrode length (mm)": getattr(self.a, "d1", np.nan) * 10,
                    "Cu foil thickness (um)": getattr(self.a_cc, "Lf", np.nan) * 1e4,
                },
                "Cathode design": {
                    "Specific capacity (mAh/g)": getattr(self.c, "Qm_am", np.nan),
                    "Active material (%)": getattr(self.c, "w_am", np.nan) * 100,
                    "Electrode width (mm)": getattr(self.c, "d0", np.nan) * 10,
                    "Electrode length (mm)": getattr(self.c, "d1", np.nan) * 10,
                    "Total # of Al-electrodes": getattr(self.c_cc, "N", np.nan),
                    "Al foil thickness (um)": getattr(self.c_cc, "Lf", np.nan) * 1e4,
                    "Active material coating wt. (mg/cm2)": getattr(
                        self.c, "Ma_am", np.nan
                    )
                    * 1e3,
                    "Porosity (%)": getattr(self.c, "por", np.nan) * 100,
                },
                "Electrolyte design": {"E/C (g/Ah)": self.EC * 1e3},
            }
            self.pretty_print_dict(out)
            # pprint(out, width=20, sort_dicts=False, indent=4)
        # elif set == "props":
        # pprint(self._attributes, width=40, sort_dicts=False, indent=4)
        elif set == "L":
            labellist = [
                "Cathode Films:",
                "Anode Films:",
                "Active Materials Sum:",
                "Al-Foils:",
                "Cu-Foils:",
                "Separators:",
                "Packaging:",
                "Inactive Materials Sum:",
                "Cell Sum:",
            ]
            nn_list = [
                getattr(self.c, "N", np.nan),
                getattr(self.a, "N", np.nan),
                1,
                getattr(self.c_cc, "N", np.nan),
                getattr(self.a_cc, "N", np.nan),
                getattr(self.s, "N", np.nan),
                2,
                1,
                1,
            ]
            ll_list = [
                getattr(self.c, "Lf", np.nan),
                getattr(self.a, "Lf", np.nan),
                getattr(self, "L_active", np.nan),
                getattr(self.c_cc, "Lf", np.nan),
                getattr(self.a_cc, "Lf", np.nan),
                getattr(self.s, "Lf", np.nan),
                getattr(self.pg, "Lf", np.nan),
                getattr(self, "L_inactive", np.nan),
                getattr(self, "L", np.nan),
            ]
            # lsum_list = [n*l for n,l in zip(nn_list,ll_list)]
            col1 = 25  # Width for the labels
            # Width for the cost per mass, adjusted for the size of the values
            col2 = 6
            col3 = 10  # Width for the mass (mg)
            col4 = 10  # Width for the total cost

            str_list = [
                # f"{lab: <25} {nn:>2.0f}x   {ll*1e4:>6.2f} um  =  {nn*ll*1e4:>7.2f} um"
                f"{lab: <{col1}}{nn:>{col2}.0f}{ll*1e4:>{col3}.0f}{nn*ll*1e4:>{col4}.0f}"
                for lab, nn, ll in zip(labellist, nn_list, ll_list)
            ]
            for i in [2, 7, 8]:
                str_list[i] = (
                    f"{labellist[i]: <{col1}}{'':>{col2}}{'':>{col3}}{ll_list[i]*1e4:>{col4}.0f}"
                )

            # replace nan with empty string
            str_list = [s.replace("nan", "   ") for s in str_list]

            print("POUCH CELL THICKNESSES:")
            print(f"{'N': >{col1+col2}}{'um': >{col3}}{'um': >{col4}}")
            for i, line in enumerate(str_list):
                if i in [0, 3, 8]:
                    print("-" * (col1 + col2 + col3 + col4))  # Underline the last row
                print(line)
            print("-" * (col1 + col2 + col3 + col4))  # Underline the last row

        elif set == "M":
            labellist = [
                "AM in Cathode Films:",
                "AM in Anode Films:",
                "Electrolyte:",
                "Active Materials Sum:",
            ]
            nn_list = [getattr(self.c, "N", np.nan), getattr(self.a, "N", np.nan), 1, 1]
            mm_list = [
                getattr(self.c, "Mf_am", np.nan),
                getattr(self.a, "Mf_am", np.nan),
                getattr(self.el, "M", np.nan),
                getattr(self, "M_active", np.nan),
            ]

            # inactive_materials
            if self.c.N_comp > 2:
                for i in range(1, self.c.N_comp):
                    labellist.append(f"CB({i}) In Cathode Films:")
                    mm_list.append(getattr(self.c, f"Mf_in_{i-1}", np.nan))
                    nn_list.append(getattr(self.c, "N", np.nan))
            elif self.c.N_comp == 1:
                labellist.append("CB In Cathode Films:")
                mm_list.append(getattr(self.c, "Mf_in", np.nan))
                nn_list.append(getattr(self.c, "N", np.nan))

            if self.a.N_comp > 2:
                for i in range(1, self.a.N_comp):
                    labellist.append(f"CB({i}) In Anode Films:")
                    mm_list.append(getattr(self.a, f"Mf_in_{i-1}", np.nan))
                    nn_list.append(getattr(self.a, "N", np.nan))
            elif self.a.N_comp == 1:
                labellist.append("CB In Anode Films:")
                mm_list.append(getattr(self.a, "Mf_in", np.nan))
                nn_list.append(getattr(self.a, "N", np.nan))

            labellist += [
                "Al-Foils:",
                "Cu-Foils:",
                "Separators:",
                "Packaging:",
                "Inactive Materials Sum:",
                "Cell Sum:",
            ]
            nn_list += [
                getattr(self.c_cc, "N", np.nan),
                getattr(self.a_cc, "N", np.nan),
                getattr(self.s, "N", np.nan),
                1,
                1,
                1,
            ]
            mm_list += [
                getattr(self.c_cc, "Mf", np.nan),
                getattr(self.a_cc, "Mf", np.nan),
                getattr(self.s, "Mf", np.nan),
                getattr(self.pg, "M", np.nan),
                getattr(self, "M_inactive", np.nan),
                getattr(self, "M", np.nan),
            ]
            col1 = 25  # Width for the labels
            # Width for the cost per mass, adjusted for the size of the values
            col2 = 6
            col3 = 10  # Width for the mass (mg)
            col4 = 10  # Width for the total cost

            str_list = [
                f"{lab: <{col1}}{nn:>{col2}.0f}{mm*1e3:>{col3}.0f}{nn*mm:>{col4}.2f}"
                for lab, nn, mm in zip(labellist, nn_list, mm_list)
            ]

            for i in [3, 10, 11]:
                str_list[i] = (
                    f"{labellist[i]: <{col1}}{'':>{col2}}{'':>{col3}}{mm_list[i]:>{col4}.2f}"
                )
                # str_list[i] = f"{labellist[i]: <47} {mm_list[i]:>7.2f} g"

            str_list = [s.replace("nan", "   ") for s in str_list]

            print("POUCH CELL MASSES:")
            print(f"{'N': >{col1+col2}}{'mg': >{col3}}{'g': >{col4}}")
            for i, line in enumerate(str_list):
                if i in [0, 4, 11, 12]:
                    print("-" * (col1 + col2 + col3 + col4))  # Underline the last row
                print(line)
            print("-" * (col1 + col2 + col3 + col4))  # Underline the last row

        elif set == "D":
            # printing the dollar amounts of the components
            # Im printing the cost per mass, the mass, and the total cost
            # Active materials
            labellist = [
                "AM in Cathode Films:",
                "AM in Anode Films:",
                "Electrolyte:",
                "Active Materials Sum:",
            ]
            dm_list = [
                getattr(self.c, "Dm_am", np.nan),
                getattr(self.a, "Dm_am", np.nan),
                getattr(self.el, "Dm", np.nan),
                1,
            ]
            mm_list = [
                getattr(self.c, "M_am", np.nan),
                getattr(self.a, "M_am", np.nan),
                getattr(self.el, "M", np.nan),
                getattr(self, "M_active", np.nan),
            ]
            dd_list = [
                getattr(self.c, "D_am", np.nan),
                getattr(self.a, "D_am", np.nan),
                getattr(self.el, "D", np.nan),
                getattr(self, "D_active", np.nan),
            ]

            # inactive_materials
            if self.c.N_comp > 2:
                for i in range(1, self.c.N_comp):
                    labellist.append(f"CB({i}) In Cathode Films:")
                    mm_list.append(getattr(self.c, f"M_in_{i-1}", np.nan))
                    dd_list.append(getattr(self.c, f"D_in_{i-1}", np.nan))
                    dm_list.append(getattr(self.c, f"Dm_in_{i-1}", np.nan))
            elif self.c.N_comp == 1:
                labellist.append("CB In Cathode Films:")
                mm_list.append(getattr(self.c, "M_in", np.nan))
                dd_list.append(getattr(self.c, "D_in", np.nan))
                dm_list.append(getattr(self.c, "Dm_in", np.nan))

            if self.a.N_comp > 2:
                for i in range(1, self.a.N_comp):
                    labellist.append(f"CB({i}) In Anode Films:")
                    mm_list.append(getattr(self.a, f"M_in_{i-1}", np.nan))
                    dd_list.append(getattr(self.a, f"D_in_{i-1}", np.nan))
                    dm_list.append(getattr(self.a, f"Dm_in_{i-1}", np.nan))
            elif self.a.N_comp == 1:
                labellist.append("CB In Anode Films:")
                mm_list.append(getattr(self.a, "M_in", np.nan))
                dd_list.append(getattr(self.a, "D_in", np.nan))
                dm_list.append(getattr(self.a, "Dm_in", np.nan))

            # add inactive materials and sum
            labellist += [
                "Al-Foils:",
                "Cu-Foils:",
                "Separators:",
                "Packaging:",
                "Inactive Materials Sum:",
                "Cell Sum:",
            ]
            # get the masses of the components
            mm_list += [
                getattr(self.c_cc, "M", np.nan),
                getattr(self.a_cc, "M", np.nan),
                getattr(self.s, "M", np.nan),
                getattr(self.pg, "M", np.nan),
                getattr(self, "M_inactive", np.nan),
                getattr(self, "M", np.nan),
            ]
            dm_list += [
                getattr(self.c_cc, "Dm", np.nan),
                getattr(self.a_cc, "Dm", np.nan),
                getattr(self.s, "Dm", np.nan),
                getattr(self.pg, "Dm", np.nan),
                1,
                1,
                1,
            ]
            dd_list += [
                getattr(self.c_cc, "D", np.nan),
                getattr(self.a_cc, "D", np.nan),
                getattr(self.s, "D", np.nan),
                getattr(self.pg, "D", np.nan),
                getattr(self, "D_inactive", np.nan),
                getattr(self, "D", np.nan),
            ]
            col1 = 25  # Width for the labels
            # Width for the cost per mass, adjusted for the size of the values
            col2 = 6
            col3 = 10  # Width for the mass (mg)
            col4 = 10  # Width for the total cost

            str_list = [
                f"{lab: <{col1}}{dm:>{col2}.2f}{mm*1e3:>{col3}.0f}{dd:>{col4}.2f}"
                for lab, dm, mm, dd in zip(labellist, dm_list, mm_list, dd_list)
            ]

            # Adjust specific rows like row 3 and 10
            for i in [3, 10, 11]:
                str_list[i] = (
                    f"{labellist[i]: <{col1}}{'':>{col2}}{mm_list[i]*1e3:>{col3}.0f}{dd_list[i]:>{col4}.2f}"
                )

            # replace nan with empty string
            str_list = [s.replace("nan", "   ") for s in str_list]

            # Print the result
            print("POUCH CELL MATERIAL COSTS:")
            print(f"{'$/kg': >{col1+col2}}{'mg': >{col3}}{'$': >{col4}}")
            # print("-" * (col1 + col2 + col3 + col4))  # Underline the header
            for i, line in enumerate(str_list):
                if i in [0, 4, 11]:
                    print("-" * (col1 + col2 + col3 + col4))
                print(line)
            print("-" * (col1 + col2 + col3 + col4))  # Underline the last row

        return

        def print_attributes(self):
            """
            Prints all attributes with color codes based on their state:
            - Green for initially given attributes
            - Blue for derived attributes (properties)
            - Red for unsolved attributes (None)
            """
            for attr, value in self.__dict__.items():
                # If it's an attribute from _attributes (initially given)
                if attr in self._attributes:
                    status = "initial"
                    color = Fore.GREEN  # Initially given: Green
                # If it's a derived attribute (property)
                elif hasattr(self, attr) and callable(getattr(self, attr)):
                    status = "derived"
                    color = Fore.BLUE  # Derived: Blue
                # If the attribute is unsolved (None or unresolved)
                elif value is None:
                    status = "unsolved"
                    color = Fore.RED  # Unsolved: Red
                else:
                    continue  # Skip non-attributes or other handled cases

                # Print the attribute name, its value, and its status with appropriate color
                print(f"{color}{attr} = {value} ({status}){Style.RESET_ALL}")
