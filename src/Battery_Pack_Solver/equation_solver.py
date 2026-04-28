# Battery_Pack_Solver/equation_solver.py
import numpy as np


def solve_equation_attr(equation, rtol=1e-5, atol=1e-8):
    """
    Solve an equation of the form
    A + B + C + D .... = 0   (sum)
    or
    A * B * C * D ... = 1   (prod)

    using object attributes.

    If only one value is unknown, it will be solved and updated on the object. returns 'solved' and the solved variable.
    If all values are known, it checks consistency and returns 'consistent' or 'inconsistent'.
    If more than one value is unknown, it returns 'unsolved'.

    Returns:
        str: 'consistent', 'inconsistent', 'solved', or 'unsolved'
        object: the variable that was solved, or None if not applicable.
    """
    # Retrieve values from the objects using getattr()
    # Filter out the known values (those that are not None)
    #
    # terms can contain constants, they have the format ('c', val)

    # print(equation)
    type = equation[0]

    def invert(val):
        if val is None:
            return None
        if type == "sum":
            return -val
        elif type == "product":
            return 1 / val if val != 0 else np.nan  # Avoid division by zero

    def combine(values):
        if type == "sum":
            sum = np.sum([val for val in values if val is not None])
            return sum
        elif type == "product":
            product = np.prod([val for val in values if val is not None])
            return product

    # first we get the rhs and lhs values
    rhs_vals = []
    rhs_terms = []
    lhs_vals = []
    lhs_terms = []
    for obj, attr, sign in equation[1:]:
        if sign == "+":
            side_v = lhs_vals
            side_t = lhs_terms
        elif sign == "-":
            side_v = rhs_vals
            side_t = rhs_terms

        if obj == "const":
            side_v.append(attr)  # If it's a constant, use its value directly
            side_t.append("const")
        else:
            side_v.append(getattr(obj, attr, None))
            side_t.append((obj, attr))

    lhs_unknowns = lhs_vals.count(None)
    rhs_unknowns = rhs_vals.count(None)

    values = lhs_vals + rhs_vals
    terms = lhs_terms + rhs_terms
    num_unknowns = lhs_unknowns + rhs_unknowns

    # if (obj, "w_in") in terms:
    # print("hello")
    # print(f"{terms = }")
    # print(f"{values = }")
    # then we check if they are all known.
    # if they are, we can check consistency
    if num_unknowns == 0:
        lhs = combine(lhs_vals)
        rhs = combine(rhs_vals)
        if np.isclose(lhs, rhs, rtol=rtol, atol=atol):
            return (
                "consistent",
                None,
                num_unknowns,
            )  # All terms are known and consistent
        else:
            return (
                "inconsistent",
                None,
                num_unknowns,
            )  # All terms are known but inconsistent

    # special case:
    # if its a product with more than one unknown
    # and
    #   the rhs has exactly one unknown, and no zeros
    # and
    #   the lhs has at least one zero
    # then
    #   the unknown is zero
    #
    # and vice versa

    if (type == "product") and (num_unknowns > 1):
        if (rhs_unknowns == 1) & (rhs_vals.count(0) == 0) & (lhs_vals.count(0) > 0):
            unknown_term = rhs_terms[rhs_vals.index(None)]
            setattr(*unknown_term, 0)
            return "solved", unknown_term, num_unknowns

        if (lhs_unknowns == 1) & (lhs_vals.count(0) == 0) & (rhs_vals.count(0) > 0):
            unknown_term = lhs_terms[lhs_vals.index(None)]
            setattr(*unknown_term, 0)
            return "solved", unknown_term, num_unknowns

    # if exactly one term is unknown, we can solve it
    if num_unknowns == 1:
        unknown_term = terms[values.index(None)]  # The only unknown term
        # True if the unknown term is on the lhs
        lhs_unknown = unknown_term in lhs_terms

        lhs = combine(lhs_vals)
        rhs = combine(rhs_vals)

        # if its a product
        if type == "product":
            # and both sides are zero,
            if (rhs == 0) & (lhs == 0):
                # the unknown remains unsolved
                return "unsolved", None, num_unknowns

            # if rhs is zero and the unknown is on the lhs
            # the unknown is zero
            if (rhs == 0) & (lhs_unknown):
                setattr(*unknown_term, 0)
                return "solved", unknown_term, num_unknowns

            # if lhs is zero and the unknown is on the rhs
            # the unknown is zero
            if (lhs == 0) & (not lhs_unknown):
                setattr(*unknown_term, 0)
                return "solved", unknown_term, num_unknowns

        # now that the zero cases are handled, the rest can be solved
        if lhs_unknown:
            result = combine([rhs, invert(lhs)])
        else:
            result = combine([lhs, invert(rhs)])

        # debugging a term here
        # if unknown_term[1] == "A":

        # print(f"{unknown_term = }")
        # print("debugging Ma")
        # print(f"{equation = }")
        # print(f"{values = }")
        # print(f"{lhs_vals = }")
        # print(f"{lhs = }")
        # print(f"{rhs_vals = }")
        # print(f"{rhs = }")
        # print(f"{result = }")
        # print(f"{combine(lhs_vals) = }")
        # print(f"{combine(rhs_vals) = }")
        # print(f"{getattr(unknown_term[0],'d0',None) = }")
        # print("-----")

        # we cant set a constant

        setattr(*unknown_term, result)
        return "solved", unknown_term, num_unknowns  # Return the solved term

    # If more than one term is missing,
    # and neither side is zero
    # it's unsolved
    return "unsolved", None, num_unknowns
