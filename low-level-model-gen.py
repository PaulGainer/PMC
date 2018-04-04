#!/usr/bin/env python3
################################################################################
#     - low level model generation script for oscillator model
################################################################################

import itertools
import math
import numpy
from scipy.special import binom
import operator
import sys
import os


'''
################################################################################
# MICAz Wireless Measurement System
################################################################################
VOLTAGE = 3.0		# 2.7V - 3.3V
DRAW_IDLE_AMPS = 0.000020
DRAW_TRANSMIT_AMPS = 0.011
DRAW_RECEIVE_AMPS = 0.0197
CYCLE_LENGTH_SECONDS = 10.0
################################################################################
'''

state_file = None
transition_file = None

#===============================================================================
# evol(x)
#===============================================================================
def evolution(x):
	return x + 1
#===============================================================================
# pert(x)
#===============================================================================
def perturbation(t, x, epsilon, alpha):
	return x * epsilon * alpha

#-------------------------------------------------------------------------------
# value used to represent \star in failure vector
#-------------------------------------------------------------------------------
F_VEC_NON_FIRED = -1

#-------------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------------
write_file = None

#-------------------------------------------------------------------------------
# directory for generated models
#-------------------------------------------------------------------------------
MODELS_DIR = 'models'

#===============================================================================
# update function
#===============================================================================
def update(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire):
	alpha_phi = alpha(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire)
	if phi <= rp:
		next_phi = evolution(phi)
	else:
		next_phi = evolution(phi) + int(round_half_up(perturbation(t, phi, epsilon, alpha_phi)))
	return (next_phi, alpha_phi)

#===============================================================================
# fire predicate
#===============================================================================
def fire(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire):
	next_phi, alpha_phi = update(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire)
	return (next_phi > t, alpha_phi)

#===============================================================================
# alpha value calculation
#===============================================================================
def alpha(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire):
	if phi == t:
		return 0
	elif f_vec[phi] != F_VEC_NON_FIRED and last_fire: # fire(n, t, epsilon, rp, phi + 1, g_state, f_vec):
		return last_alpha + g_state[phi] - f_vec[phi]
	else:
		return last_alpha

#===============================================================================
# the transition function
#===============================================================================
def tau(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire):
	fire_phi, alpha_phi = fire(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire)
	if fire_phi:
		return (fire_phi, alpha_phi, 1)
	else:
		next_phi, alpha_phi = update(n, t, epsilon, rp, phi, g_state, f_vec, last_alpha, last_fire)
		return (fire_phi, alpha_phi, next_phi)

#===============================================================================
# the successor function
#===============================================================================
def succ(n, t, epsilon, rp, g_state, f_vec, last_alpha, last_fire):
	next_state = numpy.zeros((t,), dtype = int)
	for phi in reversed(range(t)):
		last_fire, last_alpha, updated_phi = tau(n, t, epsilon, rp, phi + 1, g_state, f_vec, last_alpha, last_fire)
		next_state[updated_phi - 1] += g_state[phi]
	return next_state

f_vec_count = 1
num_global_firing_states = 0
global_firing_state = 1

#===============================================================================
# builds the set of possible failure vectors for a global state
#===============================================================================
def build_failure_vectors(n, t, epsilon, rp, phi, g_state, partial_f_vec, last_alpha, last_fire):
	global f_vec_count
	if not partial_f_vec:
		f_vec_count = 1
	f_vec_list = []
	f_vec_phi_g_1_f = [F_VEC_NON_FIRED] * phi
	f_vec_phi_g_1_f.extend(partial_f_vec)
	fire_phi, alpha_phi = fire(n, t, epsilon, rp, phi, g_state, f_vec_phi_g_1_f, last_alpha, last_fire)
	if phi > 1 and fire_phi:
		for k in range(g_state[phi - 1] + 1):
			f_vec_prefix_k = [k]
			f_vec_prefix_k.extend(partial_f_vec)
			f_vec_list.extend(build_failure_vectors(n, t, epsilon, rp, phi - 1, g_state, f_vec_prefix_k, alpha_phi, fire_phi))
	elif phi == 1:
		f_vec_prefix_non_fired = [F_VEC_NON_FIRED]
		f_vec_prefix_non_fired.extend(partial_f_vec)
		fire_phi, alpha_phi = fire(n, t, epsilon, rp, 1, g_state, f_vec_prefix_non_fired, last_alpha, last_fire)
		if fire_phi:
			for k in range(g_state[0] + 1):
				f_vec_prefix_k = [k]
				f_vec_prefix_k.extend(partial_f_vec)
				f_vec_list.extend([tuple(f_vec_prefix_k)])
				f_vec_count += 1				
	if not fire_phi:
		f_vec = [F_VEC_NON_FIRED] * phi
		f_vec.extend(partial_f_vec)
		f_vec_list.extend([tuple(f_vec)])
		f_vec_count += 1				
	return f_vec_list

#===============================================================================
# build the low level model
#===============================================================================
def build_low_level_model(n, t, epsilon, rp):
	# build state map
	INITIAL_STATE_NUM = 0
	next_state_num = INITIAL_STATE_NUM + 1
	global_states = []
	state_map = dict()
	transition_list = []
	for state in combinations_with_replacement_counts(n - 1, t):
		state_list = list(state)
		state_list[t - 1] += 1
		global_states.append(tuple(state_list))
	firing_states = [state for state in global_states if state[t - 1] > 0]
	# add firing states to state map
	state_map[str(tuple(numpy.zeros(t, dtype = int)))] = INITIAL_STATE_NUM
	for state in firing_states:
		state_map[str(state)] = next_state_num
		next_state_num += 1
	binomial_products = []
	# precalculate binomial coefficient products
	for state in firing_states:
		binomial_product = 1
		sigma_n_i = 0
		for i in range(t):
			binomial_product *= binom(n - sigma_n_i, state[i])
			sigma_n_i = sigma_n_i + state[i]				
		binomial_products.append(binomial_product)
	binomial_product_sum = sum(binomial_products)
	# add initial transitions for the low level model to the transition list
	it_count = 0
	for state in firing_states:
		it_count = it_count + 1
		state_array = numpy.array(state)
		multiplier = numpy.argmax(state_array > 0) + 1
		transition_list.append((INITIAL_STATE_NUM, state_map[str(state)], '({}/(pow({},{})))'.format(multiplier * binomial_products[it_count - 1], t, n)))		
	# add transitions to successor states of firing states	
	for state in firing_states:
		this_state_num = state_map[str(state)]
		successor_states = []
		low_pr_string_list = []
		for f_vec in build_failure_vectors(n, t, epsilon, rp, t, state, [], 0, True):
			successor_state = succ(n, t, epsilon, rp, state, f_vec, 0, True)
			index = [i for i, j in enumerate(successor_states) if numpy.array_equal(successor_state, j)]
			if not index:
				successor_states.append(successor_state)
				low_pr_string_list.append(low_level_p_tau(n, t, epsilon, rp, state, f_vec))
			else:
				low_pr_string_list[index[0]] += ('+%s' % low_level_p_tau(n, t, epsilon, rp, state, f_vec))
		for i in range(len(successor_states)):
			successor_state_str = str(tuple(successor_states[i]))
			successor_state_num = None
			if successor_state_str in state_map:
				successor_state_num = state_map[successor_state_str]
			else:
				successor_state_num = next_state_num
				next_state_num += 1
				global_states.append(tuple(successor_state))
				state_map[successor_state_str] = successor_state_num
			transition_list.append((this_state_num, successor_state_num, '({})'.format(low_pr_string_list[i])))
	# build transitions from non-firing states to successor firing states
	non_firing_states = [state for state in global_states if state[t - 1] == 0]
	for state in non_firing_states:
		state_array = numpy.array(state)
		delta = t - 1 - numpy.nonzero(state_array)[0][-1]
		state_array = numpy.roll(state, delta)
		transition_list.append((state_map[str(state)], state_map[str(tuple(state_array))], 1.0))
	write_output(t, state_map, transition_list)
	
#===============================================================================
# write states and transitions to files
#===============================================================================
def write_output(t, state_map, transition_list):
	set_write_file(state_file)
	first_line = '('
	for i in range(t - 1):
		first_line += 'k_{},'.format(i + 1)
	first_line += 'k_{})'.format(t)
	writeln(first_line)
	sorted_states = sorted(state_map.items(), key = operator.itemgetter(1))
	for state, state_num in sorted_states:
		writeln('{}:{}'.format(state_num, state.replace(' ', '')))
	state_file.close()
	
	set_write_file(transition_file)
	first_line = '{} {}'.format(len(sorted_states), len(transition_list))	
	writeln(first_line)
	sorted_transitions = sorted(transition_list, key = lambda x: (x[0], x[1]))
	for item in sorted_transitions:
		writeln('{} {} {}'.format(item[0], item[1], item[2]))
	transition_file.close()
	
#===============================================================================
# low level p_fail
#===============================================================================
def low_level_p_fail(n, b):
	expression = ''
	if b == 1:
		expression += 'MU'
	elif b != 0:
		expression += 'pow({},{})'.format('MU', b)
	if n - b == 1:
		expression += '{}(1-MU)'.format('*' if expression else '')
	elif n - b != 0:
		expression += '{}pow(1-{},{})'.format('*' if expression else '', 'MU', n - b)
	binom_n_b = binom(n, b)
	if binom_n_b != 1:
		expression += '{}{}'.format('*' if expression else '', binom_n_b)
	return expression
#	return ('pow({},{})*pow(1-{},{})*{}'.format('MU', b, 'MU', n - b, binom(n, b)))

#===============================================================================
# low level p_tau	
#===============================================================================
def low_level_p_tau(n, t, epsilon, rp, g_state, f_vec):
	product = ''
	is_one = False
	for i in range(t):
		p_fail_string = None
		if f_vec[i] != F_VEC_NON_FIRED:
			p_fail_string = low_level_p_fail(g_state[i], f_vec[i])
			product += p_fail_string
			is_one = True
		if i < t - 1 and is_one:
			if p_fail_string:
				product += '*'
	if not is_one:
		product += '1'
	return product

#===============================================================================
# return all permutations of k unlaballed balls into n boxes
#===============================================================================	
def combinations_with_replacement_counts(k, n):
   size = n + k - 1
   for indices in itertools.combinations(range(size), n - 1):
       starts = [0] + [index + 1 for index in indices]
       stops = indices + (size,)
       yield tuple(map(operator.sub, stops, starts))

#===============================================================================
# set the file to write to
#===============================================================================
def set_write_file(f):
	global write_file
	write_file = f

#===============================================================================
# write a line to the active file, then new line
#===============================================================================	
def writeln(string, tabs = 0):
	write(string + '\n', tabs = tabs)

#===============================================================================
# write a line to the active file, no new line
#===============================================================================	
def write(string, tabs = 0):
	write_file.write((tabs * '\t') + string)
	
#===============================================================================
# override default python3 rounding and always round 0.5 up
#===============================================================================
def round_half_up(x):
	return math.ceil(x) if x - math.floor(x) >= 0.5 else math.floor(x)

#===============================================================================
# entry point
#===============================================================================
if __name__ == '__main__':
	if(len(sys.argv) != 5):
		print('usage: ./low-level-model-gen.py <N> <T> <epsilon> <R>')
	else:
		N = sys.argv[1]
		T = sys.argv[2]
		EPSILON = sys.argv[3]
		RP = sys.argv[4]
		if not os.path.exists(MODELS_DIR):
		    os.makedirs(MODELS_DIR)
		state_file = open('{}/{}_{}_{}_{}.sta'.format(MODELS_DIR, N, T, EPSILON, RP), 'w')
		transition_file = open('{}/{}_{}_{}_{}.tra'.format(MODELS_DIR, N, T, EPSILON, RP), 'w')
		build_low_level_model(int(N), int(T), float(EPSILON), int(RP))
	
	



