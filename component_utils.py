# Battery_Pack_Solver/component_utilities.py
import re


# Function to remove color codes
def remove_color_codes(text):
    # Remove ANSI escape sequences using a regex
    return re.sub(r"\033\[[0-9;]*m", "", text)
