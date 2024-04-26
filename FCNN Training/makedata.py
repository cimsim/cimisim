import csv
import numpy as np
import matplotlib.pyplot as plt

def _find_nearest(array, value):
    """
    Helper function for makedata()
    Finds the nearest value in an array to a given value.

    Parameters:
    - array: An array to search within.
    - value: The target value.

    Returns:
    - nearest_value: The nearest value in the array to the target value.
    - idx: The index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def findClosest(arr, target):
    """
    Binary search to find the closest value in a sorted array to a target value.

    Parameters:
    - arr: A sorted array.
    - target: The target value to find the closest value to.

    Returns:
    - closest_value: The closest value in the array to the target value.
    """
    n = len(arr)

    # Corner cases
    if (target <= arr[0]):
        return arr[0]
    if (target >= arr[n - 1]):
        return arr[n - 1]

    # Doing binary search
    i = 0
    j = n
    mid = 0
    while (i < j):
        mid = (i + j) // 2

        if (arr[mid] == target):
            return arr[mid]

        # If target is less than array element, then search in the left half
        if (target < arr[mid]):

            # If target is greater than the previous element to mid, return the closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], target)

            # Repeat for the left half
            j = mid

        # If target is greater than mid
        else:
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], target)

            # Update i
            i = mid + 1

    # Only a single element left after the search
    return arr[mid]

def find_closestsort_indx(array, value):
    """
    Finds the index of the closest value to the given value in a sorted array.

    Parameters:
    - array: A sorted array.
    - value: The target value.

    Returns:
    - idx: The index of the closest value in the array.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx

def find_closest_indx(array, value):
    """
    Finds the index of the closest value to the given value in an array.

    Parameters:
    - array: An array.
    - value: The target value.

    Returns:
    - idx: The index of the closest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def getClosest(val1, val2, target):
    """
    Compares two values and returns the one closest to a target value.

    Parameters:
    - val1: The first value.
    - val2: The second value.
    - target: The target value.

    Returns:
    - closest_value: The closest value to the target value.
    """
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1

def gen_value_array(ini=0, fin=1, A_P=0.5, A_D=-0.5, N=16):
    """
    Generates an array of values following a specific pattern.

    Parameters:
    - ini: Initial value.
    - fin: Final value.
    - A_P: Parameter LTP.
    - A_D: Parameter LTD.
    - N: Number of values to generate.

    Returns:
    - G: The generated array of values.
    - G_P: The positive part of the generated array.
    - G_D: The negative part of the generated array.
    """
    Gmax = fin
    Gmin = ini
    print("AP,AD", A_P, A_D)
    n_max = 2**(N)
    n = np.arange(0, n_max, 1) / n_max
    B = (Gmax - Gmin) / (1 - np.exp(-1 / A_P))
    G_P = B * (1 - np.exp(-n / A_P)) + Gmin
    G_D = -((Gmax - Gmin) * (1 - np.exp((1 - n) / A_D)) /
            (1 - np.exp((1) / A_D))) + Gmax
    G_D = np.flip(G_D)
    G = np.concatenate((G_P, G_D))
    return G, G_P, G_D

def _find_nearest(array, value):
    """
    Helper function for makedata()
    Finds the nearest value in an array to a given value.

    Parameters:
    - array: An array to search within.
    - value: The target value.

    Returns:
    - nearest_value: The nearest value in the array to the target value.
    - idx: The index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx



def makedata_double_2(N_msb=5, N_lsb=4, A_P=0.5, A_D=-0.5):

    # Creating dictionaries to map weight values to their increment/decrement values
    pow2N_msb = int(2**N_msb)
    pow2N_msb2 = int(pow2N_msb / 2)
    pow2N_lsb = int(2**N_lsb)
    pow2N_lsb2 = int(pow2N_lsb / 2)
    increasing_weights_msb = {}
    decreasing_weights_msb = {}
    increasing_weights_lsb = {}
    decreasing_weights_lsb = {}

    # Generate weights for MSB and LSB
    weights_msb = gen_value_array(ini=-5, fin=5, A_P=A_P, A_D=A_D, N=N_msb)[0]
    weights_lsb = gen_value_array(ini=-5 / pow2N_lsb, fin=5 / pow2N_lsb, A_P=A_P, A_D=A_D, N=N_lsb)[0]

    potentiation_weights_msb = weights_msb[:pow2N_msb2]
    depression_weights_msb = weights_msb[pow2N_msb2:]
    potentiation_weights_lsb = weights_lsb[:pow2N_lsb2]
    depression_weights_lsb = weights_lsb[pow2N_lsb2:]

    # Create mappings for increasing weights (MSB and LSB)
    for index, weight in enumerate(potentiation_weights_msb[:-1]):
        increasing_weights_msb[weight] = potentiation_weights_msb[index + 1] - weight
    increasing_weights_msb[potentiation_weights_msb[-1]] = 0

    for index, weight in enumerate(depression_weights_msb):
        intermediate_weight, intermediate_index = _find_nearest(potentiation_weights_msb, weight)
        if potentiation_weights_msb[intermediate_index] != potentiation_weights_msb[-1]:
            increasing_weights_msb[weight] = potentiation_weights_msb[intermediate_index + 1] - weight
        else:
            increasing_weights_msb[weight] = 0

    for index, weight in enumerate(depression_weights_msb[:-1]):
        decreasing_weights_msb[weight] = depression_weights_msb[index + 1] - weight
    decreasing_weights_msb[depression_weights_msb[-1]] = 0

    # Create mappings for increasing weights (LSB)
    for index, weight in enumerate(potentiation_weights_lsb[:-1]):
        increasing_weights_lsb[weight] = potentiation_weights_lsb[index + 1] - weight
    increasing_weights_lsb[potentiation_weights_lsb[-1]] = 0

    for index, weight in enumerate(depression_weights_lsb):
        intermediate_weight, intermediate_index = _find_nearest(potentiation_weights_lsb, weight)
        if potentiation_weights_lsb[intermediate_index] != potentiation_weights_lsb[-1]:
            increasing_weights_lsb[weight] = potentiation_weights_lsb[intermediate_index + 1] - weight
        else:
            increasing_weights_lsb[weight] = 0

    for index, weight in enumerate(depression_weights_lsb[:-1]):
        decreasing_weights_lsb[weight] = depression_weights_lsb[index + 1] - weight
    decreasing_weights_lsb[depression_weights_lsb[-1]] = 0

    # Combine mappings for MSB and LSB
    increasing_weights = (increasing_weights_msb, increasing_weights_lsb)
    decreasing_weights = (decreasing_weights_msb, decreasing_weights_lsb)

    # Combine weights from both parts
    weights_actual = []
    for msb in weights_msb:
        for lsb in weights_lsb:
            weights_actual.append(msb + lsb)
    weights = np.array(weights_actual)

    # Save weights to a file and plot the data
    print(len(weights))
    print(np.sort(weights))
    np.savetxt('./weights.txt', weights)
    pulses = np.arange(0, len(weights))
    plt.plot(pulses, weights)
    plt.xlabel('Pulse number')
    plt.ylabel('Weight')
    plt.savefig('weightdata.png')
    stddev = 0
    return weights, increasing_weights, decreasing_weights, stddev
