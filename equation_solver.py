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
