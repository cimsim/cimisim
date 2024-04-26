# Purpose: Maps a weight into its real synaptic conductance that will be stored on the crossbar.
#Called by: CNN_inference_only.py

import numpy as np
from math import exp, log, floor, ceil


#Helper function for mapping a weight to its synaptic value
def find_nearest_helper(a, ap, ad, g_min, g_max, p_max):
    is_pot = True


    bp = (g_max - g_min)/(1-exp(-1/ap))
   
    p_analytic = -p_max*ap*log(1 - (a-g_min)/bp)
    min_diff = abs(a - (bp*(1-exp(-ceil(p_analytic)/(ap*p_max))) + g_min))
    p_match = ceil(p_analytic)
    if abs(a - (bp*(1-exp(-floor(p_analytic)/(ap*p_max))) + g_min)) < min_diff:
        min_diff = abs(a - (bp*(1-exp(-floor(p_analytic)/(ap*p_max))) + g_min))
        p_match = floor(p_analytic)
       
    bd = (g_max - g_min)/(1-exp(-1/ad))
    p_analytic = p_max*(ad*log(1 + (a - g_max)/bd) + 1)
    if abs(a - (-bd*(1-exp((floor(p_analytic)/p_max - 1)/ad)) + g_max)) < min_diff:
        min_diff = abs(a - (-bd*(1-exp((floor(p_analytic)/p_max - 1)/ad)) + g_max))
        p_match = floor(p_analytic)
        is_pot = False
    if abs(a - (-bd*(1-exp((ceil(p_analytic)/p_max - 1)/ad)) + g_max)) < min_diff:
        min_diff = abs(a - (-bd*(1-exp((ceil(p_analytic)/p_max - 1)/ad)) + g_max))
        p_match = ceil(p_analytic)
        is_pot = False
   
    if p_match < 0:
        p_match = 0
    if p_match > p_max:
        p_match = p_max


    if is_pot is True:
        return (bp*(1-exp(-p_match/(ap*p_max))) + g_min), p_match
    else:
        return (-bd*(1-exp((p_match/p_max - 1)/ad)) + g_max), p_max+p_match


#Wrapper for the helper function
def find_nearest_v2(a, ap, ad, g_min, g_max, p_max, dv_set, dv):
    b = np.copy(a)
    num_pulses = np.zeros(shape = a.shape)
    for idx, j in np.ndenumerate(a):
        b[idx], num_pulses[idx] = find_nearest_helper(j, ap, ad, g_min, g_max, p_max)
    if dv_set == 'Multiplicative':
        b*= (1+np.random.normal(loc = 0.0, scale = dv, size = b.shape)) #Multiplicative noise
    elif dv_set == 'Additive':
        b += np.random.normal(loc = 0.0, scale = dv*(g_max), size = b.shape) #Additive noise
    return b, num_pulses  




