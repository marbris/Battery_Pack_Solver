# Battery_Pack_Solver/component.py
import numpy as np
from colorama import Fore, Style
from .equation_solver import solve_equation_attr
from .component_utils import remove_color_codes
import scipy.optimize


class Component:
    def __init__(self, attributes):
        self._attributes = attributes
        self.reset_attributes()  # Initialize attributes from the provided dictionary
        self.tolerance = 1e-6  # Allowable relative error when checking consistency
        self.products = []  # List to hold product equations
        self.sums = []  # List to hold sum equations

    def reset_attributes(self):
        # Iterate over all attributes in __dict__ and delete them
        for attr in list(self.__dict__.keys()):
            if attr != "_attributes":  # Don't clear _attributes
                delattr(self, attr)  # Delete the attribute

        # Re-add all attributes from self._attributes
        for key, value in self._attributes.items():
            setattr(self, key, value)

    def solve_equation(self, eq, equation_type=None):
        """
        Solve an equation using the external solver function and update the respective lists.
        """

        status, solved_var, num_unknowns = solve_equation_attr(eq, type=equation_type)

        return status, solved_var, num_unknowns

    def solve_component(self, verbose=False):
        """
        Iteratively solve all possible values and check consistency.
        If nothing is changed, it returns False
        """
        self.inconsistent = []  # Store any inconsistent equations
        # Track variables that are involved in equations but not solved yet
        self.unsolved_variables = set()
        self.equations = []  # List to store equation details

        if verbose == "vv":
            print("#" * (len(self.Description) + len("Solving ") + 2 * (1 + 7)))
            print(f"####### Solving {self.Description} #######")
            print("#" * (len(self.Description) + len("Solving ") + 2 * (1 + 7)))

        # Create a list of tuples (eq_type, eq) for both products and sums
        equations = []
        # Add product equations with eq_type 'product'
        if self.products:
            for eq in self.products:
                equations.append(("product", eq))
        # Add sum equations with eq_type 'sum'
        if self.sums:
            for eq in self.sums:
                equations.append(("sum", eq))

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
            for eq_type, eq in equations:
                # Solve the equation
                status, solved_var, num_unknowns = self.solve_equation(
                    eq, equation_type=eq_type
                )

                equation_str = self.get_equation_string(eq, eq_type, color_output=True)

                if solved_var is not None:
                    solved_var = f"{solved_var[0].desc}.{solved_var[1]}"

                # Add equation info to the list
                self.equations.append(
                    {
                        "equation": eq,
                        "equation_str": equation_str,
                        "equation_type": eq_type,
                        "num_terms": len(eq),
                        "num_unknowns": num_unknowns,
                        "status": status,
                    }
                )
                if status == "inconsistent":
                    self.inconsistent.append(
                        (eq, (self._attributes[var] for var in eq))
                    )
                    if verbose == "vv" or verbose == "v":
                        print(f"{equation_str:<60} is inconsistent")

                if status == "consistent":
                    if verbose == "vv":
                        print(f"{equation_str:<60} is consistent")

                if status == "unsolved":
                    # add the unsolved variables to the set
                    unsolved_vars = (var for var in eq if var not in self._attributes)
                    self.unsolved_variables.update(unsolved_vars)

                    if verbose == "vv":
                        print(f"{equation_str:<60} unsolved")

                if status == "solved":
                    # Remove solved variables from unsolved set
                    self.unsolved_variables.difference_update(eq)
                    solved = True
                    anything_solved = True

                    if verbose == "vv" or verbose == "v":
                        print(f"{equation_str:<40} solved for {solved_var:>15}")

        return anything_solved  # Return True if any variable was solved

    def get_attribute(self, term, color_output=False):
        # Check if its a constant
        obj, attr = term
        if obj == "c":
            term_str = f"{term[1]}"
            color = Fore.WHITE  # Constants: White (same as operators)
            value = attr
            status = "constant"  # Status for constants
        else:
            if attr in obj._attributes:
                status = "initial"
                color = Fore.GREEN  # Initially given: Green
                value = obj._attributes[attr]
            # If it's a derived attribute (property)
            elif hasattr(obj, attr):
                status = "derived"
                color = Fore.BLUE  # Derived: Blue
                value = getattr(obj, attr)  # Call the property method
            else:
                status = "unsolved"
                color = Fore.RED  # Unsolved: Red
                value = None

            term_str = f"{obj.desc}.{attr}"
        if color_output:
            term_str = f"{color}{term_str}{Style.RESET_ALL}"

        return obj, attr, value, status, term_str

    # get list of all attributes in the list of equations
    def get_attribute_info(self, ignore_constants=True, color_output=False):
        attribute_info = set()
        equations = self.products + self.sums
        for eq in equations:
            for term in eq:
                obj, attr, value, status, term_str = self.get_attribute(
                    term, color_output=color_output
                )
                if not (obj == "c" and ignore_constants):
                    attribute_info.add(
                        (obj, attr, value, status, term_str)
                    )  # Add the description of the object

        return attribute_info

    # print attributes of the component
    # color them based on whether they are solved or not
    def print_attributes(self, getstring=False):
        attributes_info = self.get_attribute_info(color_output=False)
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

    def get_equation_string(self, eq, eq_type, color_output=False):
        """
        Generate an equation string for given terms, applying colors based on their status
        (initial, derived, unsolved).
        """
        # operator string
        if eq_type == "product":
            operator = " * "
        elif eq_type == "sum":
            operator = " + "

        # Build the equation string
        termlist = []
        for term in eq:
            _, _, _, _, term_str = self.get_attribute(term, color_output=color_output)
            termlist.append(term_str)

        left_side = operator.join(termlist[:-1])  # All except the last term
        right_side = termlist[-1]  # The last term
        equation_str = f"{left_side} = {right_side}"

        return equation_str

    def print_equations(self, getstring=False):
        """
        Sort equations first by status (consistent/unsolved/inconsistent), then by the
        number of variables that are "initial", "derived", and "unsolved".
        """
        # Get the attribute info for each equation
        # attribute_info = self.get_attribute_info()

        # Helper function to count variables by status
        def count_variable_status(eq):
            initial_count = 0
            derived_count = 0
            unsolved_count = 0

            # Go through each term in the equation
            for obj, attr in eq:
                if obj != "c":  # Exclude constants
                    if attr in self._attributes:
                        initial_count -= 1
                    elif hasattr(self, attr):
                        derived_count -= 1
                    else:
                        unsolved_count -= 1

            return (
                initial_count,
                derived_count,
                unsolved_count,
            )

        # Sort equations based on their status and the count of "initial", "derived", and "unsolved"
        self.equations.sort(
            key=lambda eq: (
                # Sorting by status: 'inconsistent' > 'consistent' > 'unsolved'
                {"inconsistent": 0, "consistent": 1, "unsolved": 2}[eq["status"]],
                # Sorting by number of initial, derived, and unsolved variables
                *count_variable_status(eq["equation"]),
            )
        )

        max_eq_len_header = np.max(
            [len(remove_color_codes(eq["equation_str"])) for eq in self.equations]
        )
        max_status_len_header = np.max(
            [len(remove_color_codes(eq["status"])) for eq in self.equations]
        )
        max_eq_len = np.max([len(eq["equation_str"]) for eq in self.equations])
        max_status_len = np.max([len(eq["status"]) for eq in self.equations])
        # max_status_len = max(len(status_header), max(len(remove_color_codes(row.split()[-1])) for row in table_body))
        max_eq_len_header += 5
        max_status_len_header += 5
        max_eq_len += 5
        max_status_len += 5

        # Prepare the result string
        result = ""
        result += (
            f"{'Equations':<{max_eq_len_header}} {'Status':<{max_status_len_header}}\n"
        )
        result += "-" * (max_eq_len_header + max_status_len_header) + "\n"  # Divider

        for eq in self.equations:
            equation_str = eq["equation_str"]
            status = eq["status"]

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
            result += f"{equation_str:<{max_eq_len}} {color}{status.capitalize():<{max_status_len}}{Style.RESET_ALL}\n"
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
        self.products = []  # List to hold product equations
        self.sums = []  # List to hold sum equations

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
                w_sum_terms.append((self, f"w_{label}"))
                self.products += [
                    # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                    (
                        (self, "M"),
                        (self, f"w_{label}"),
                        (self, f"M_{label}"),
                    ),
                ]

            # adding the equation for the sum of all component ratios
            # sum wi = 1
            self.sums.append((*w_sum_terms, ("c", 1)))
            # if there are more than two components,
            # w_in is the sum of the inactive materials:
            if N_comp > 2:
                self.sums.append((*w_sum_terms[1:], (self, "w_in")))

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

                    # the equation is:
                    # I need the Volume of component i per electrode mass
                    # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                    self.products += [
                        # (Mass i/Vol i)*(Vol i/Mass c) = (Mass i/Mass c)
                        (
                            (self, f"Mv_{label}"),
                            (self, f"Vm_{label}"),
                            (self, f"w_{label}"),
                        ),
                    ]
                    # self.products.append(
                    # (
                    # (self, f"Mv_{label}"),
                    # (self, f"Vm_{label}"),
                    # (self, f"w_{label}"),
                    # )
                    # )
                    v_sum_terms.append((self, f"Vm_{label}"))

                # sum (Vol i/Mass c) = (Vol solid phase/Mass c)
                self.sums.append((*v_sum_terms, (self, "Vm_solid")))
                # (Mass c/ Vol solid phase) * (Vol solid phase/Mass c) = 1
                self.products.append(((self, "Mv_solid"), (self, "Vm_solid"), ("c", 1)))

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

                    # (Cost i/Mass i)*(Mass i/Mass c)*(Mass c/film) = (Cost i/film)
                    self.products += [
                        (
                            (self, f"Dm_{label}"),
                            (self, f"w_{label}"),
                            (self, "Mf"),
                            (self, f"Df_{label}"),
                        ),
                        (
                            (self, f"Df_{label}"),
                            (self, "N"),
                            (self, f"D_{label}"),
                        ),
                    ]
                    d_sum_terms.append((self, f"Df_{label}"))

                # sum of all component costs per film = total cost of film
                self.sums.append((*d_sum_terms, (self, "Df")))

        self.products += [
            # Area of Film
            ((self, "d0"), (self, "d1"), (self, "A")),
            # Volume of Film
            ((self, "Lf"), (self, "A"), (self, "Vf")),
            # Pore Volume of Film
            ((self, "Vf"), (self, "por"), (self, "Vf_por")),
            # Solid Phase Volume of Film
            ((self, "Vf"), (self, "por_inv"), (self, "Vf_solid")),
            # (1-por) * Solid Phase Density = film density
            ((self, "por_inv"), (self, "Mv_solid"), (self, "Mv")),
            # film Volume * film Density =  film Mass
            ((self, "Vf"), (self, "Mv"), (self, "Mf")),
            # Mass of Film * Active Material Ratio = Active Material Mass
            ((self, "Mf"), (self, "w_am"), (self, "Mf_am")),
            # Mass of Film * Inactive Material Ratio = Inactive Material Mass
            ((self, "Mf"), (self, "w_in"), (self, "Mf_in")),
            # Areal Mass of Film * Area = Total Mass of Electrode
            ((self, "Ma"), (self, "A"), (self, "Mf")),
            # Film Volume * Film Density = Film Mass
            ((self, "Vf"), (self, "Mv"), (self, "Mf")),
            # Active Material Specific Capacity * Active Material Mass = Film Capacity
            ((self, "Qm_am"), (self, "Mf_am"), (self, "Qf")),
            # Areal Capacity * Area = Film Capacity
            ((self, "Qa"), (self, "A"), (self, "Qf")),
            # Average Voltage * capacity = Specific Energy
            ((self, "Vavg"), (self, "Qm_am"), (self, "Em_am")),
            # Active Material Mass * Active Material Specific Capacity = Film Capacity
            ((self, "Qm_am"), (self, "Mf_am"), (self, "Qf")),
            # Active Material Mass * Active Material Specific Capacity = Film Capacity
            ((self, "Em_am"), (self, "Mf_am"), (self, "Ef")),
            # Mass of Film * Number of Films = Total Mass of Electrode
            ((self, "Mf"), (self, "N"), (self, "M")),
            # Mass of Cathode * Active Mass ratio = Active Material Mass of Electrode
            ((self, "M"), (self, "w_am"), (self, "M_am")),
            # Areal Mass of Electrode * Active Mass ratio = Areal Active Mass Material Mass of Electrode
            ((self, "Ma"), (self, "w_am"), (self, "Ma_am")),
            # Mass of Cathode * Inactive Mass ratio = Inactive Material Mass of Electrode
            ((self, "M"), (self, "w_in"), (self, "M_in")),
            # Active Material Mass of Film * Number of Films = Total Active Material Mass of Electrode
            ((self, "Mf_am"), (self, "N"), (self, "M_am")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            ((self, "Mf_in"), (self, "N"), (self, "M_in")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            ((self, "Mf_in"), (self, "N"), (self, "M_in")),
            # Film Thickness * Number of Films = Total Thickness of Electrode
            ((self, "Lf"), (self, "N"), (self, "L")),
            # Film Volume * Number of Films = Total Volume of Electrode
            ((self, "Vf"), (self, "N"), (self, "V")),
            # Film Pore Volume * Number of Films = Total Pore Volume of Electrode
            ((self, "Vf_por"), (self, "N"), (self, "V_por")),
            # Film Capacity * Number of Films = Total Capacity of Electrode
            ((self, "Qf"), (self, "N"), (self, "Q")),
            # Film Energy * Number of Films = Total Energy of Electrode
            ((self, "Ef"), (self, "N"), (self, "E")),
            # Active Material Ratio AM_ratio = w_am/w_in
            # ((self, "AM_ratio"), (self, "w_in"), (self, "w_am")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ((self, "Dm_am"), (self, "M_am"), (self, "D_am")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ((self, "Dm_am"), (self, "Mf_am"), (self, "Df_am")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ((self, "Dm"), (self, "Mf"), (self, "Df")),
            # Film Cost * Number of Films = Electrode Cost
            ((self, "Df"), (self, "N"), (self, "D")),
            # Inactive Material Mass of Film * Number of Films = Total Inactive Material Mass of Electrode
            # (Vol in) * (Mass in / Vol in) = Mass in
            ((self, "V_in"), (self, "Mv_in"), (self, "M_in")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ((self, "V_am"), (self, "Mv_am"), (self, "M_am")),
        ]
        self.sums += [
            # porosity + inverse porosity = 1, useful for density calculations
            ((self, "por"), (self, "por_inv"), ("c", 1)),
            # active material cost + inactive material cost = total material cost
            ((self, "Df_am"), (self, "Df_in"), (self, "Df")),
            # active material cost + inactive material cost = total material cost
            ((self, "D_am"), (self, "D_in"), (self, "D")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ((self, "V_am"), (self, "V_in"), (self, "V")),
            # (Vol in) * (Mass in / Vol in) = Mass in
            ((self, "M_am"), (self, "M_in"), (self, "M")),
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

    def initialize(self):

        # equations A * B = C
        self.products = [
            # side1 * side2 = Area
            ((self, "d0"), (self, "d1"), (self, "A")),
            # Film Thickness * Area = Film Volume
            ((self, "Lf"), (self, "A"), (self, "Vf")),
            # Areal Mass of Film * Area = Total Mass of Film
            ((self, "Ma"), (self, "A"), (self, "Mf")),
            # Film Volume * Film Density = Film Mass
            ((self, "Vf"), (self, "Mv"), (self, "Mf")),
            # Mass of Film * Number of Films = Total Mass of Electrode
            ((self, "Mf"), (self, "N"), (self, "M")),
            # Film Thickness * Number of Films = Total Thickness of Electrode
            ((self, "Lf"), (self, "N"), (self, "L")),
            # Electrode Material cost per Electrode Mass * Film mass = Film Cost
            ((self, "Dm"), (self, "Mf"), (self, "Df")),
            # Film Cost * Number of Films = Electrode Cost
            ((self, "Df"), (self, "N"), (self, "D")),
            # Film Volume * Porosity = Film Pore Volume
            ((self, "Vf"), (self, "por"), (self, "Vf_por")),
            # Solid Phase Volume of Film
            ((self, "Vf"), (self, "por_inv"), (self, "Vf_solid")),
            # (1-por) * Solid Phase Density = film density
            ((self, "por_inv"), (self, "Mv_solid"), (self, "Mv")),
            # Film Pore Volume * Number of Films = Total Pore Volume of Electrode
            ((self, "Vf_por"), (self, "N"), (self, "V_por")),
        ]
        # self.sums = []
        self.sums = [
            # Inverse porosity, useful for density calculations
            ((self, "por"), (self, "por_inv"), ("c", 1)),
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

    def initialize(self):
        # equations A * B = C
        self.products = [
            # Volume, Density, Mass relationship
            ((self, "V"), (self, "Mv"), (self, "M")),
            # Cost/Mass, Mass, Cost
            ((self, "Dm"), (self, "M"), (self, "D")),
        ]
        self.sums = []

        # self.solve_all()  # Solve and validate
        # self.solve_component()  # Solve and validate

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

        if "d0_c" in self._attributes and "d1_c" in self._attributes:
            self.c._attributes["d0"] = self._attributes["d0_c"]
            self.c._attributes["d1"] = self._attributes["d1_c"]
            self.c_cc._attributes["d0"] = self._attributes["d0_c"]
            self.c_cc._attributes["d1"] = self._attributes["d1_c"]

        if "d0_a" in self._attributes and "d1_a" in self._attributes:
            self.a._attributes["d0"] = self._attributes["d0_a"]
            self.a._attributes["d1"] = self._attributes["d1_a"]
            self.a_cc._attributes["d0"] = self._attributes["d0_a"]
            self.a_cc._attributes["d1"] = self._attributes["d1_a"]

        # equations A * B * C * D ... = N
        self.products = [
            # anode equations
            # Capacity ratio NP of anode and cathode
            # mAh_a/mAh_c, ratio between the capacities of cathode and anode
            # mAh_c/film , specific capacity of cathode film
            # mAh_a/film , specific capacity of anode film
            ((self, "NP"), (self.c, "Qf"), (self.a, "Qf")),
            # mAh_a/g_aam, specific capacity of anode active material
            # (mAh_c/g_cam)/(mAh_a/g_aam), ratio between the specific capacities of cathode and anode active materials
            # mAh_c/g_cam, specific capacity of cathode active material
            ((self.a, "Qm_am"), (self, "Qm_am_ratio"), (self.c, "Qm_am")),
            # mAh_a/mAh_c, ratio between the capacities of cathode and anode
            # (mAh_c/g_cam)/(mAh_a/g_aam), ratio between the specific capacities of cathode and anode active materials
            # g_aam/g_cam, amount of anode active material per cathode active material
            ((self, "NP"), (self, "Qm_am_ratio"), (self, "NP_cam")),
            # g_aam/g_cam, amount of anode active material per cathode active material
            # g_cam, mass active cathode material
            # g_aam/g_cam, amount of anode active material per cathode active material
            ((self, "NP_cam"), (self.c, "M_am"), (self.a, "M_am")),
            # Electrolyte equations
            # Amount of electrolyte in cell by EC and capacity
            ((self, "EC"), (self.c, "Q"), (self.el, "M")),
            # mAh_c/g_cam, specific capacity of cathode active material
            # g_el/mAh_c, amount of electrolyte in cell per capacity of active cathode
            # g_el/g_cam, amount of electrolyte in cell per mass of active cathode material
            ((self.c, "Qm_am"), (self, "EC"), (self, "EC_cam")),
            # g_el/g_cam, amount of electrolyte in cell per mass of active cathode material
            # g_cam, mass active cathode material
            # g_el, mass of electrolyte in cell
            ((self, "EC_cam"), (self.c, "M_am"), (self.el, "M")),
            # g_am/g_cam, Total Active Mass of Cell per active cathode material mass
            # g_cam, mass active cathode material
            # g_am, mass of active material in cell
            ((self, "M_active_cam"), (self.c, "M_am"), (self, "M_active")),
            # energy density of cell
            ((self, "Em"), (self, "M"), (self, "E")),
            # Capacity of cell
            ((self, "Qm"), (self, "M"), (self, "Q")),
            # Area * Thickness = Volume
            ((self, "A"), (self, "L"), (self, "V")),
            # Active Material material ratio of cell
            ((self, "M"), (self, "M_am_ratio"), (self, "M_active")),
            # Total Energy in pouch cell
            ((self, "Q"), (self, "Vavg"), (self, "E")),
            # Total energy per volume in pouch cell
            ((self, "Ev"), (self, "V"), (self, "E")),
            # energy density in the limit of no inactive materials
            ((self, "Em_active"), (self, "M_active"), (self, "E")),
        ]

        # equations A + B + C + ... - N = 0
        self.sums = [
            # Total Mass of Cell
            (
                (self.el, "M"),
                (self.c, "M"),
                (self.a, "M"),
                (self.a_cc, "M"),
                (self.c_cc, "M"),
                (self.s, "M"),
                (self.pg, "M"),
                (self, "M"),
            ),
            # Total Active Mass of Cell
            (
                (self.el, "M"),
                (self.c, "M_am"),
                (self.a, "M_am"),
                (self, "M_active"),
            ),
            # Total Inactive Mass of Cell
            (
                (self.c, "M_in"),
                (self.a, "M_in"),
                (self.a_cc, "M"),
                (self.c_cc, "M"),
                (self.s, "M"),
                (self.pg, "M"),
                (self, "M_inactive"),
            ),
            # Total Mass of Cell
            (
                (self, "M_inactive"),
                (self, "M_active"),
                (self, "M"),
            ),
            # Total Inactive Mass in Electrodes
            (
                (self.c, "M_in"),
                (self.a, "M_in"),
                (self, "M_ac_inactive"),
            ),
            # Total Inactive Mass in excluding Electrodes
            (
                (self.a_cc, "M"),
                (self.c_cc, "M"),
                (self.s, "M"),
                (self.pg, "M"),
                (self, "M_not_ac_inactive"),
            ),
            # Total Inactive Mass of Cell
            (
                (self, "M_ac_inactive"),
                (self, "M_not_ac_inactive"),
                (self, "M_inactive"),
            ),
            # Total Cost of Cell
            (
                (self.el, "D"),
                (self.c, "D"),
                (self.a, "D"),
                (self.a_cc, "D"),
                (self.c_cc, "D"),
                (self.s, "D"),
                (self.pg, "D"),
                (self, "D"),
            ),
            # Cost of active materials in Cell
            (
                (self.el, "D"),
                (self.c, "D_am"),
                (self.a, "D_am"),
                (self, "D_active"),
            ),
            # Total Inactive Mass of Cell
            (
                (self.c, "D_in"),
                (self.a, "D_in"),
                (self.a_cc, "D"),
                (self.c_cc, "D"),
                (self.s, "D"),
                (self.pg, "D"),
                (self, "D_inactive"),
            ),
            # Total Cost of Cell
            (
                (self, "D_inactive"),
                (self, "D_active"),
                (self, "D"),
            ),
            # 1 + NP_cam + EC_cam = M_active_cam Total Active Mass of Cell per amount of cathode active material
            # g_aam/g_cam, amount of anode active material per cathode active material
            # g_aam/g_cam, amount of anode active material per cathode active material
            # g_am/g_cam, Total Active Mass of Cell per active cathode material mass
            (("c", 1), (self, "NP_cam"), (self, "EC_cam"), (self, "M_active_cam")),
            # Total inactive thickness of cell
            (
                (self.c_cc, "L"),
                (self.a_cc, "L"),
                (self.s, "L"),
                (self.pg, "L"),
                (self, "L_inactive"),
            ),
            # Total active thickness of cell
            ((self.c, "L"), (self.a, "L"), (self, "L_active")),
            # Total Pouch cell thickness
            ((self, "L_active"), (self, "L_inactive"), (self, "L")),
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
            solved |= self.a.solve_component(verbose=verbose)
            solved |= self.c.solve_component(verbose=verbose)
            solved |= self.el.solve_component(verbose=verbose)

            solved |= self.a_cc.solve_component(verbose=verbose)
            solved |= self.c_cc.solve_component(verbose=verbose)
            solved |= self.s.solve_component(verbose=verbose)
            solved |= self.pg.solve_component(verbose=verbose)

            # Solve for PouchCell itself
            solved |= self.solve_component(verbose=verbose)

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

            str_list = [
                f"{lab: <25} {nn:>2.0f}x   {ll*1e4:>6.2f} um  =  {nn*ll*1e4:>7.2f} um"
                for lab, nn, ll in zip(labellist, nn_list, ll_list)
            ]
            for i in [2, 7, 8]:
                str_list[i] = f"{labellist[i]: <45} {ll_list[i]*1e4:>7.2f} um"

            print("POUCH CELL THICKNESSES:")
            for i, line in enumerate(str_list):
                if i in [0, 3, 8]:
                    print("-" * 56)
                print(line)
            print("-" * 56)

        elif set == "M":
            labellist = [
                "AM in Cathode Films:",
                "AM in Anode Films:",
                "Electrolyte:",
                "Active Materials Sum:",
                "CB in Cathode Films:",
                "CB in Anode Films:",
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
                1,
                getattr(self.c, "N", np.nan),
                getattr(self.a, "N", np.nan),
                getattr(self.c_cc, "N", np.nan),
                getattr(self.a_cc, "N", np.nan),
                getattr(self.s, "N", np.nan),
                1,
                1,
                1,
            ]
            mm_list = [
                getattr(self.c, "Mf_am", np.nan),
                getattr(self.a, "Mf_am", np.nan),
                getattr(self.el, "M", np.nan),
                getattr(self, "M_active", np.nan),
                getattr(self.c, "Mf_in", np.nan),
                getattr(self.a, "Mf_in", np.nan),
                getattr(self.c_cc, "Mf", np.nan),
                getattr(self.a_cc, "Mf", np.nan),
                getattr(self.s, "Mf", np.nan),
                getattr(self.pg, "M", np.nan),
                getattr(self, "M_inactive", np.nan),
                getattr(self, "M", np.nan),
            ]

            str_list = [
                f"{lab: <25} {nn:>2.0f}x   {mm*1e3:>8.2f} mg  =  {nn*mm:>7.2f} g"
                for lab, nn, mm in zip(labellist, nn_list, mm_list)
            ]
            for i in [3, 10, 11]:
                str_list[i] = f"{labellist[i]: <47} {mm_list[i]:>7.2f} g"

            print("POUCH CELL MASSES:")
            for i, line in enumerate(str_list):
                if i in [0, 4, 11, 12]:
                    print("-" * 57)
                print(line)
            print("-" * 57)
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
            # i need to loop because the number of components can vary

            if self.c.w_in != 0:
                if self.c.N_comp > 2:
                    for i in range(1, self.c.N_comp):
                        labellist += (f"CB({i}) In Cathode Films:",)
                        mm_list += (getattr(self.c, f"M_in_{i-1}", np.nan),)
                        dd_list += (getattr(self.c, f"D_in_{i-1}", np.nan),)
                        dm_list += (getattr(self.c, f"Dm_in_{i-1}", np.nan),)
                else:
                    labellist += ("CB In Cathode Films:",)
                    mm_list += (getattr(self.c, "M_in", np.nan),)
                    dd_list += (getattr(self.c, "D_in", np.nan),)
                    dm_list += (getattr(self.c, "Dm_in", np.nan),)
            if self.a.w_in != 0:
                if self.a.N_comp > 2:
                    for i in range(1, self.a.N_comp):
                        labellist += (f"CB({i}) In Anode Films:",)
                        mm_list += (getattr(self.a, f"M_in_{i-1}", np.nan),)
                        dd_list += (getattr(self.a, f"D_in_{i-1}", np.nan),)
                        dm_list += (getattr(self.a, f"Dm_in_{i-1}", np.nan),)
                else:
                    labellist += ("CB In Anode Films:",)
                    mm_list += (getattr(self.a, "M_in", np.nan),)
                    dd_list += (getattr(self.a, "D_in", np.nan),)
                    dm_list += (getattr(self.a, "Dm_in", np.nan),)

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

            # Print the result
            print("POUCH CELL MATERIAL COSTS:")
            print(f"{'$/kg': >{col1+col2}}{'mg': >{col3}}{'$': >{col4}}")
            # print("-" * (col1 + col2 + col3 + col4))  # Underline the header
            for i, line in enumerate(str_list):
                if i in [0, 4, 11]:
                    print("-" * (col1 + col2 + col3 + col4))
                print(line)
            print("-" * (col1 + col2 + col3 + col4))  # Underline the last row
            # col1 = 30
            # col2 = 10
            # str_list = [
            # f"{lab: <{col1}} ${dm:>5.5f}/kg   {mm*1e3:>8.2f} mg  =  ${dd:>7.2f} "
            # for lab, dm, mm, dd in zip(labellist, dm_list, mm_list, dd_list)
            # ]
            # for i in [3, 10]:
            # str_list[i] = (
            # f"{labellist[i]: <{col1}}  {mm_list[i]*1e3:>8.2f} mg  =  ${dd_list[i]:>7.2f} "
            # )

            # # print(labellist)
            # # print(dd_list)
            # # print(dm_list)
            # # print(mm_list)
            # # print(len(labellist))
            # # print(len(dd_list))
            # # print(len(dm_list))
            # # print(len(mm_list))
            # print("POUCH CELL MATERIAL COSTS:")
            # for i, line in enumerate(str_list):
            # if i in [0, 4, 11]:
            # print("-" * 57)
            # print(line)
            # print("-" * 57)

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
