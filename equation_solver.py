# Battery_Pack_Solver/equation_solver.py
import numpy as np


def solve_equation_attr0(terms, type):
    """
    Solve an equation of the form
    A + B + C + D .... = N   (sum)
    or
    A * B * C * D ... = N   (prod)

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
    known = []
    unknown = []
    known_sum = 0
    for obj, attr in terms:
        if obj == "c":
            val = attr  # If it's a constant, use its value directly
        else:
            val = getattr(obj, attr, None)

        if val is None:
            unknown.append((obj, attr))
        else:
            known.append((obj, attr))
            if type == "sum":
                # If it's the last term, treat it as negative
                if (obj, attr) == terms[-1]:
                    known_sum -= val
                else:
                    known_sum += val
            if type == "product":
                # If it's the last term, treat it as negative
                if (obj, attr) == terms[-1]:
                    known_sum -= np.log(val)
                else:
                    known_sum += np.log(val)

    num_unknowns = len(unknown)
    # If there is exactly one unknown (None), proceed with solving the equation
    if num_unknowns == 1:

        unknown_term = unknown[0]  # The only unknown term
        if unknown_term == terms[-1]:
            # If the unknown term is the last one, treat it as negative
            known_sum = -known_sum

        # Calculate the sum of the known terms
        if type == "sum":
            # Update the unknown term with the negative sum of known terms
            setattr(*unknown_term, -known_sum)
        elif type == "product":
            # take the exponential to undo the log transformation
            # and if the value is smaller than 1e-12, i'll take it to be zero
            value = np.exp(-known_sum)
            if value <= 1e-12:
                value = 0.0  # Avoid logging zero or negative values

            # Update the unknown term with the product of known terms
            setattr(*unknown_term, value)

        return "solved", unknown_term, num_unknowns  # Return the solved term

    # Check consistency if all three values are known
    if num_unknowns == 0:
        if type == "sum":
            error = known_sum
        elif type == "product":
            error = np.exp(known_sum) - 1
            # if error <= 1e-12:
            # value = 0.0  # Avoid logging zero or negative values

        if abs(error) > 1e-6:  # Tolerance for error
            # eq_string = get_equation_string(terms, type)
            # print(eq_string)
            # print(error)
            return (
                "inconsistent",
                None,
                num_unknowns,
            )  # All terms are known but inconsistent
        else:
            return (
                "consistent",
                None,
                num_unknowns,
            )  # All terms are known and consistent

    # If more than one term is missing, it's unsolved
    return "unsolved", None, num_unknowns


def solve_equation_attr(terms, type):
    """
    Solve an equation of the form
    A + B + C + D .... = N   (sum)
    or
    A * B * C * D ... = N   (prod)

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

    def invert(val, type):
        if val is None:
            return None
        if type == "sum":
            return -val
        elif type == "product":
            return 1 / val if val != 0 else np.nan  # Avoid division by zero

    def combine(values, type):
        if type == "sum":
            return np.sum([val for val in values if val is not None])
        elif type == "product":
            return np.prod([val for val in values if val is not None])

    # first we get all the values
    values = []
    for obj, attr in terms:
        if obj == "c":
            values.append(attr)  # If it's a constant, use its value directly
        else:
            values.append(getattr(obj, attr, None))

    known = [val for val in values if val is not None]
    num_unknowns = len(values) - len(known)

    # then we check if they are all known.
    # if they are, we can check consistency
    if num_unknowns == 0:
        lhs = combine(values[:-1], type)
        rhs = values[-1]  # The last term is the right-hand side
        if np.isclose(lhs, rhs):
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
    # in a product,
    # where the last term (rhs) is unknown,
    # and where one of the terms on the left hand side is zero,
    # then the equation is solved with the last term set to zero.
    if (type == "product") & (values[-1] is None) & (0 in values[:-1]):
        setattr(*terms[-1], 0)
        return "solved", terms[-1], num_unknowns

    # if exactly one term is unknown, we can solve it
    if num_unknowns == 1:

        unknown_term = terms[values.index(None)]  # The only unknown term

        lhs = combine(values[:-1], type)
        rhs = values[-1]  # The last term is the right-hand side

        # if its a product and rhs is zero,
        if (type == "product") & (rhs == 0):
            if lhs == 0:
                # and both sides are zero,
                # it is unsolved
                return "unsolved", None, num_unknowns
            else:
                # if the rhs is zero,
                # we can set the unknown term to zero
                setattr(*unknown_term, 0)
                return "solved", unknown_term, num_unknowns

        # now that the zero case is handled, the rest can be solved
        # if the unknown term is the last one, we just combine,
        # otherwise we invert the last term, combine, and invert the result
        if unknown_term == terms[-1]:
            result = combine(values, type)

            setattr(*unknown_term, result)
            return "solved", unknown_term, num_unknowns  # Return the solved term
        else:
            # Invert the right-hand side
            values[-1] = invert(values[-1], type)
            # we combine all the (non-None) terms.
            # the remainder is the inverted unknown term
            result = combine(values, type)
            result = invert(result, type)  # Invert the result to get the unknown term
            setattr(*unknown_term, result)
        return "solved", unknown_term, num_unknowns  # Return the solved term

    # If more than one term is missing,
    # and the rhs is not zero
    # it's unsolved
    return "unsolved", None, num_unknowns


# this one has a different format for the equt
# Old format :
#   self.products += ((self, "Vavg"), (self, "Qm_am"), (self, "Em_am"))
#   self.sums += ((self, "M_am"), (self, "M_in"), (self, "M")),
# New format:
#   self.equation += ( 'prod', (self, "Vavg", '+'), (self, "Qm_am", '+'), (self, "Em_am", '-') )
#   self.equation += ( 'sum', (self, "M_am", '+'), (self, "M_in", '+'), (self, "M", '-') )
def solve_equation_attr2(equation, rtol=1e-5, atol=1e-8):
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
