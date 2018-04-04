#!/usr/bin/env python3
################################################################################
# analysis of vpmc reconfiguration
################################################################################
import collections
import itertools
import numpy
import os
import pprint
from random import random
from random import shuffle
import subprocess
import sys

AUTOMATON_TMP_FILE_PATH = 'automaton_tmp'
DAG_TMP_FILE_PATH = 'dag_tmp'
PDF_GENERATOR = 'ps2pdf'
PDF_VIEWER = 'evince'
GRAPH_LAYOUT_GENERATOR = 'dot'
COLOUR_SCHEME = 'spectral5'
NUM_COLOURS = 5
INITIAL_STATE = 'i'
ACCEPT_STATE = 'a'
next_vertex_num = 0


SYNCH_MODEL_GENERATOR = './low-level-model-gen.py'
MODEL_PATH = 'models/'


MU = 0.1

#===============================================================================
#-------------------------------------------------------------------------------
# automaton/dag output for GraphViz
#-------------------------------------------------------------------------------
#===============================================================================

def output_dag(G, file_path = None, horizontal = False, no_labels = False,
		map_tuple = None, title = None):
	vertices, edges = G
	vertex_map = edge_map = transition_map = None
	reused_vertices = list()
	if map_tuple:
		vertex_map, edge_map, transition_map = map_tuple
		for s in vertex_map:
			for label, v in vertex_map[s].items():
				reused_vertices.append(v)
	handle = open(file_path, 'w') if file_path else sys.stdout
	handle.write('digraph\n{\n')
	if horizontal:
		handle.write('\trankdir="LR";\n')
	if title:
		handle.write('\tlabel="{}";\n'.format(title))
		handle.write('\tlabelloc="t";\n')
	for label, v in vertices.items():
		colour = 'Black' if not map_tuple or v in reused_vertices else 'Red'
		if no_labels:
			label = '' if no_labels else label
		handle.write('\tnode[shape=rectangle, color={}, \
				style=rounded, label="{}"] {};\n'.format(colour, label, v))
	for v, v_p in edges:
		handle.write('\t{} -> {} [label=""];\n'.format(v, v_p))
	handle.write('}\n')
	handle.close()
	
def output_automaton(D, initial, accept, file_path = None, horizontal = False,
		no_labels = False, infected = None, title = None):
	states, pmatrix = D
	handle = open(file_path, 'w') if file_path else sys.stdout
	handle.write('digraph\n{\n')	
	if horizontal:
		handle.write('\trankdir="LR";\n')
	if title:
		handle.write('\tlabel="{}";\n'.format(title))
		handle.write('\tlabelloc="t";\n')
	for s in states:
		label = '' if no_labels else s
		shape = 'doublecircle' if eval(accept) else 'circle'
		colour = 'Red' if (infected and s in infected) else 'Black'
		handle.write('\tnode[color={}, shape={}, label="{}"] {};\n'.format(
				colour, shape, label, states[s]))
		if eval(initial):
			handle.write('\tnode[shape=point, color="Black", label=""] init_{};\n'.format(
					states[s]))
			handle.write('\tinit_{} -> {};\n'.format(states[s], states[s]))
	for s, s_p in pmatrix:
		prob = pmatrix[s, s_p]
		handle.write('\t{} -> {} [label="{}"];\n'.format(states[s], states[s_p],
				prob))
	handle.write('}')
	handle.close()	

#===============================================================================
#-------------------------------------------------------------------------------
# dag construction
#-------------------------------------------------------------------------------
#===============================================================================

def new_vertex(p, vertices):
	global next_vertex_num
	vertices[p] = next_vertex_num
	next_vertex_num += 1
	return next_vertex_num - 1

def add_vertex(p, vertices):
	if p in vertices:
		return vertices[p]
	else:
		return new_vertex(p, vertices)

def add(p1, p2):
	return '({}+{})'.format(p1 , p2)

def product(p1, p2):
	return '({}*{})'.format(p1, p2)

def geo_series(p):
	return '(1/(1-{}))'.format(p)

def add_geo_series_p_dag(p, G):
	vertices, edges = G
	geo = geo_series(p)
	if geo in vertices:
		return geo, vertices[geo]
	else:
		p_vertex = add_vertex(p, vertices)
		geo_vertex = add_vertex(geo, vertices)
		edges.append((p_vertex, geo_vertex))
		return geo, geo_vertex
	
def add_product_to_dag(p1, p2, G):
	vertices, edges = G
	pr = product(p1, p2)
	if pr in vertices:
		return pr, vertices[pr]
	else:
		p1_p2_vertex = add_vertex(pr, vertices)
		p1_vertex = add_vertex(p1, vertices)
		p2_vertex = add_vertex(p2, vertices)
		edges.append((p1_vertex, p1_p2_vertex))
		if p2 != p1:
			edges.append((p2_vertex, p1_p2_vertex))
		return pr, p1_p2_vertex
	
def add_sum_to_dag(p1, p2, G):
	vertices, edges = G
	su = add(p1, p2)
	if su in vertices:
		return su, vertices[su]
	else:
		p1_p2_vertex = add_vertex(su, vertices)
		p1_vertex = add_vertex(p1, vertices)
		p2_vertex = add_vertex(p2, vertices)
		edges.append((p1_vertex, p1_p2_vertex))
		if p2 != p1:		
			edges.append((p2_vertex, p1_p2_vertex))
		return su, p1_p2_vertex

def build_dag(D, initial, accept, order, infected, no_labels = False, horizontal = False,
		visualise = False):
	D_states, D_pmatrix = D
	states = dict(D_states)
	pmatrix = dict(D_pmatrix)
	vertices = dict()
	edges = list()
	vertex_map = dict()
	edge_map = dict()
	transition_map = collections.defaultdict(list)
	dag_viewer = None
	automaton_viewer = None
	num_products = 0
	num_sums = 0
	num_geos = 0
	if visualise:
		run_process(PDF_GENERATOR, [DAG_TMP_FILE_PATH])
		run_process(PDF_GENERATOR, [AUTOMATON_TMP_FILE_PATH])		
		dag_viewer = run_process_no_output(
				PDF_VIEWER, ['{}.pdf'.format(DAG_TMP_FILE_PATH)])
		automaton_viewer = run_process_no_output(
				PDF_VIEWER, ['{}.pdf'.format(AUTOMATON_TMP_FILE_PATH)])
		output_automaton((states, pmatrix), initial, accept,
				file_path = AUTOMATON_TMP_FILE_PATH, no_labels = no_labels,
				horizontal = horizontal, infected = infected)
		run_process(GRAPH_LAYOUT_GENERATOR,
				[AUTOMATON_TMP_FILE_PATH, '-Tpdf', '-O'])
		print('elimination order: ', end = '')
		print(order)
	for elim_state in order:
		elim_state_num = states[elim_state]
		if visualise:
			input('next elimination state: \'{}\':{}'.format(elim_state,
					elim_state_num))
		states, pmatrix, new_vertices, new_edges, transition_map, num_products, num_sums, num_geos = \
				eliminate_state(elim_state, states, pmatrix, vertices, edges,
						transition_map, infected,
						num_products, num_sums, num_geos)
		vertex_map[elim_state] = \
				dict(set(new_vertices.items()) - set(vertices.items()))
		edge_map[elim_state] = list(set(new_edges) - set(edges))
		vertices = new_vertices
		edges = new_edges
		if visualise:
			output_dag((vertices, edges), file_path = DAG_TMP_FILE_PATH,
					no_labels = no_labels, horizontal = horizontal)
			run_process(GRAPH_LAYOUT_GENERATOR, [DAG_TMP_FILE_PATH,
					'-Tpdf', '-O'])
			output_automaton((states, pmatrix), initial, accept,
					file_path = AUTOMATON_TMP_FILE_PATH, no_labels = no_labels,
					horizontal = horizontal, infected = infected)
			run_process(GRAPH_LAYOUT_GENERATOR, [AUTOMATON_TMP_FILE_PATH,
					'-Tpdf', '-O'])
	if visualise:
		input('all states eliminated')
		run_process('rm', [DAG_TMP_FILE_PATH])
		run_process('rm', ['{}.pdf'.format(DAG_TMP_FILE_PATH)])
		run_process('rm', [AUTOMATON_TMP_FILE_PATH])
		run_process('rm', ['{}.pdf'.format(AUTOMATON_TMP_FILE_PATH)])
		dag_viewer.kill()
		automaton_viewer.kill()
	return (states, pmatrix), (vertices, edges), (vertex_map, edge_map,
			transition_map), num_products, num_sums, num_geos

def eliminate_state(elim_state, in_states, in_pmatrix, in_vertices, in_edges,
		in_transition_map, infected, num_products, num_sums, num_geos):
	states = dict(in_states)
	pmatrix = dict(in_pmatrix)
	vertices = dict(in_vertices)
	edges = list(in_edges)
	transition_map = in_transition_map.copy()
	pre_states = get_pre_states(elim_state, in_states, pmatrix,
			include_self_loops = False)
	post_states = get_post_states(elim_state, in_states, pmatrix,
			include_self_loops = False)
	for pre in pre_states:
		for post in post_states:
			label = pmatrix[pre, elim_state]
			vertex = add_vertex(label, vertices)
			if (elim_state, elim_state) in pmatrix:
				loop_label = pmatrix[elim_state, elim_state]
				geo_series_sum, geo_vertex = add_geo_series_p_dag(loop_label,
						(vertices, edges))
				num_geos += 1
				label, vertex = add_product_to_dag(label, geo_series_sum,
						(vertices, edges))
				num_products += 1
			label, vertex = add_product_to_dag(label, pmatrix[elim_state, post],
					(vertices, edges))
			num_products += 1
			old_label = label
			if (pre, post) in pmatrix:
				label, vertex = add_sum_to_dag(pmatrix[pre, post], label,
						(vertices, edges))
				num_sums += 1
			if elim_state not in infected and \
					((pre in infected and (post in infected or post == ACCEPT_STATE)) 
					or (post in infected and (pre in infected or pre == INITIAL_STATE))
					or (pre == INITIAL_STATE and post == ACCEPT_STATE)):
				transition_map[pre, post].append(old_label)
			pmatrix[pre, post] = label
	pmatrix = {(s, s_p):prob
			for (s, s_p), prob in pmatrix.items()
			if s != elim_state and s_p != elim_state}
	del states[elim_state]
	return states, pmatrix, vertices, edges, transition_map, num_products, num_sums, num_geos

def eliminate_state_rebuild(elim_state, in_states, in_pmatrix, in_vertices, in_edges,
		transition_map, infected, num_products, num_sums, num_geos):
	states = dict(in_states)
	pmatrix = dict(in_pmatrix)
	vertices = dict(in_vertices)
	edges = list(in_edges)
	pre_states = get_pre_states(elim_state, in_states, pmatrix,
			include_self_loops = False)
	post_states = get_post_states(elim_state, in_states, pmatrix,
			include_self_loops = False)
	for pre in pre_states:
		for post in post_states:
			label = pmatrix[pre, elim_state]
			vertex = add_vertex(label, vertices)
			if (elim_state, elim_state) in pmatrix:
				loop_label = pmatrix[elim_state, elim_state]
				geo_series_sum, geo_vertex = add_geo_series_p_dag(loop_label,
						(vertices, edges))
				num_geos += 1
				label, vertex = add_product_to_dag(label, geo_series_sum,
						(vertices, edges))
				num_products += 1
			label, vertex = add_product_to_dag(label, pmatrix[elim_state, post],
					(vertices, edges))
			num_products += 1
			old_label = label
			if (pre, post) in pmatrix:
				label, vertex = add_sum_to_dag(pmatrix[pre, post], label,
						(vertices, edges))
				num_sums += 1
			pmatrix[pre, post] = label
	pmatrix = {(s, s_p):prob
			for (s, s_p), prob in pmatrix.items()
			if s != elim_state and s_p != elim_state}
	del states[elim_state]
	return states, pmatrix, vertices, edges, transition_map, num_products, num_sums, num_geos
	
def rebuild_dag(D, D_p, map_tuple, initial, accept, order, no_labels = False,
		horizontal = False, visualise = False,
		display_reusability_results = False):
	in_states_p, in_pmatrix_p = D_p
	states, pmatrix = D
	states_p = dict(in_states_p)
	pmatrix_p = dict(in_pmatrix_p)
	vertices_p = dict()
	edges_p = list()
	vertex_map, edge_map, transition_map = map_tuple
	# compute the initial infected set
	infected = compute_infected(D, D_p)
	reused_vertices = 0
	reused_edges = 0
	dag_viewer = None
	automaton_viewer = None
	if visualise:
		run_process(PDF_GENERATOR, [DAG_TMP_FILE_PATH])
		run_process(PDF_GENERATOR, [AUTOMATON_TMP_FILE_PATH])		
		dag_viewer = run_process_no_output(PDF_VIEWER,
				['{}.pdf'.format(DAG_TMP_FILE_PATH)])
		automaton_viewer = run_process_no_output(PDF_VIEWER,
				['{}.pdf'.format(AUTOMATON_TMP_FILE_PATH)])
		output_automaton((states_p, pmatrix_p), initial, accept,
				file_path = AUTOMATON_TMP_FILE_PATH, no_labels = no_labels,
				horizontal = True, infected = infected)
		run_process(GRAPH_LAYOUT_GENERATOR, [AUTOMATON_TMP_FILE_PATH,
				'-Tpdf', '-O'])
	#for elim_state in order:
	first_infected_state = True
	num_products = 0
	num_sums = 0
	num_geos = 0
	while len(order) > 0:
		elim_state = order[0]
		order = order[1:]
		elim_state_num = states_p[elim_state]
		if visualise:
			input('next elimination state: \'{}\':{}'.format(elim_state,
					elim_state_num))
		neighbourhood = get_pre_states(elim_state, states_p, pmatrix_p)
		neighbourhood = neighbourhood.union(get_post_states(elim_state,
				states_p, pmatrix_p))		
		if elim_state not in infected:
			mapped_vertices = vertex_map[elim_state]
			mapped_edges = edge_map[elim_state]
			reused_vertices += len(mapped_vertices)
			reused_edges += len(mapped_edges)
			vertices_p = {**vertices_p, **mapped_vertices}
			edges_p.extend(mapped_edges)
			pmatrix_p = {(s, s_p):prob
					for (s, s_p), prob in pmatrix_p.items()
					if s != elim_state and s_p != elim_state}								
			del states_p[elim_state]
		else:
			if first_infected_state:
				first_infected_state = not first_infected_state
				if (INITIAL_STATE, ACCEPT_STATE) in transition_map and len(transition_map[INITIAL_STATE, ACCEPT_STATE]) > 0:
					label = transition_map[INITIAL_STATE, ACCEPT_STATE].pop(0)
					pmatrix_p[INITIAL_STATE, ACCEPT_STATE] = label
			for s, s_p in ((s, s_p) for s, s_p in transition_map
					if (s == elim_state or s_p == elim_state)
					and len(transition_map[s, s_p]) > 0):
				if (s, s_p) in pmatrix_p:
					transition_map[s, s_p].insert(0, pmatrix_p[s, s_p])
				label = transition_map[s, s_p].pop(0)
				for l in transition_map[s, s_p]:
					label, vertex = add_sum_to_dag(l, label, (vertices_p, edges_p))
					#label = add(label, l)
				pmatrix_p[s, s_p] = label
			# this is the last state to be eliminated so remember to include
			# any label from the initial state to the accepting state
			for (s, s_p) in transition_map.copy():
				if s == elim_state or s_p == elim_state:
					del transition_map[s, s_p]
			states_p, pmatrix_p, vertices_p, edges_p, transition_map, num_products, num_sums, num_geos = \
					eliminate_state_rebuild(elim_state, states_p, pmatrix_p, vertices_p,
							edges_p, transition_map, infected, num_products, num_sums, num_geos)
			infected = infected.union(neighbourhood)
		if visualise:
			title = 'N:{}/{}, E:{}/{}'.format(reused_vertices, len(vertices_p),
					reused_edges, len(edges_p))
			output_dag((vertices_p, edges_p), file_path = DAG_TMP_FILE_PATH,
					no_labels = no_labels, horizontal = False,
					map_tuple = map_tuple, title = title)
			run_process(GRAPH_LAYOUT_GENERATOR, [DAG_TMP_FILE_PATH,
					'-Tpdf', '-O'])
			output_automaton((states_p, pmatrix_p), initial, accept,
					file_path = AUTOMATON_TMP_FILE_PATH, no_labels = no_labels,
					horizontal = True, infected = infected)
			run_process(GRAPH_LAYOUT_GENERATOR, [AUTOMATON_TMP_FILE_PATH,
					'-Tpdf', '-O'])
	if visualise:
		input('all states eliminated')
		run_process('rm', [DAG_TMP_FILE_PATH])
		run_process('rm', ['{}.pdf'.format(DAG_TMP_FILE_PATH)])
		run_process('rm', [AUTOMATON_TMP_FILE_PATH])
		run_process('rm', ['{}.pdf'.format(AUTOMATON_TMP_FILE_PATH)])
		dag_viewer.kill()
		automaton_viewer.kill()
	if display_reusability_results:
		percent_vertex_reuse = reused_vertices / len(vertices_p) * 100
		percent_edge_reuse = reused_edges / len(edges_p) * 100
		print('reused {0:0.1f}% of vertices and {0:0.1f}% of edges'.format(
				percent_vertex_reuse, percent_edge_reuse))
	return (states_p, pmatrix_p), (vertices_p, edges_p), (vertex_map, edge_map,
			transition_map), num_products, num_sums, num_geos

#===============================================================================
#-------------------------------------------------------------------------------
# model importing
#-------------------------------------------------------------------------------
#===============================================================================

def get_automaton_from_files(state_file_path, pmatrix_file_path):
	global next_vertex_num0
	if os.path.isfile(state_file_path) and os.path.isfile(pmatrix_file_path):		
		state_file = open(state_file_path, 'r')
		pmatrix_file = open(pmatrix_file_path, 'r')
		file_states = generate_states(state_file)
		file_pmatrix = generate_pmatrix(pmatrix_file, file_states)
		state_file.close()
		pmatrix_file.close()
		return file_states, file_pmatrix
		
def generate_states(state_file):
	states = dict()
	for line in state_file.readlines():
		tokens = line.rstrip().split(':')
		if len(tokens) == 2:
			states[tokens[1]] = int(tokens[0])
	return states
	
def generate_pmatrix(pmatrix_file, states):
	reverse_states = {v: k for k, v in states.items()}
	pmatrix = dict()
	for line in pmatrix_file.readlines():
		tokens = line.rstrip().split(' ')
		if len(tokens) == 3:
			pmatrix[reverse_states[int(tokens[0])],
					reverse_states[int(tokens[1])]] = tokens[2]	
	return pmatrix

#===============================================================================
#-------------------------------------------------------------------------------
# graph operations
#-------------------------------------------------------------------------------
#===============================================================================

def change_structure(D, cull_edges = [], add_edges = [], add_states = []):
	states, pmatrix = D
	for state in add_states:
		states.add(state)
	pmatrix = {(s, s_p): pmatrix[s, s_p]
			for s, s_p in pmatrix
			if (s, s_p) not in cull_edges}
	for s, s_p, label in add_edges:
		pmatrix[s, s_p] = label
	return states, pmatrix

def compute_infected(D, D_p):
	states, pmatrix = D
	states_p, pmatrix_p = D_p
	infected = set()
	for s, s_p in pmatrix:
		if (s, s_p) not in pmatrix_p or \
				pmatrix[s, s_p] != pmatrix_p[s, s_p]:
			infected.add(s)
			infected.add(s_p)
	for s, s_p in pmatrix_p:
		if (s, s_p) not in pmatrix or \
				pmatrix[s, s_p] != pmatrix_p[s, s_p]:
			infected.add(s)
			infected.add(s_p)
	return infected
	
#-------------------------------------------------------------------------------
# compute all states from which the accepting states0 are reachable
#-------------------------------------------------------------------------------
def remove_p_zero_states(D, accept):
	states, pmatrix = D
	states_p = dict(states)
	pmatrix_p = dict(pmatrix)
	subgraph = list(s for s, s_num in states_p.items() if eval(accept))
	expanded = True
	while(expanded):
		new_states = list()
		for s, s_p in pmatrix_p:
			if s_p in subgraph and s not in new_states and s not in subgraph:
				new_states.append(s)
		subgraph.extend(new_states)
		expanded = len(new_states) > 0
	states_p = {state: s_num
			for state, s_num in states_p.items()
			if state in subgraph}
	pmatrix_p = {(s, s_p): pmatrix_p[s, s_p]
			for s, s_p in pmatrix_p
			if s in subgraph and s_p in subgraph}
	return (states_p, pmatrix_p)

def make_accepting_states_absorbing(D, accept):
	states, pmatrix = D
	states_p = dict(states)
	pmatrix_p = dict(pmatrix)
	subgraph = list(s for s, s_num in states_p.items() if eval(accept))
	pmatrix_p = {(s, s_p): pmatrix_p[s, s_p]
			for (s, s_p) in pmatrix_p
			if s not in subgraph}
	return (states_p, pmatrix_p)
	
def insert_init_and_accept(D, initial, accept):
	states, pmatrix = D
	states_p = dict(states)
	pmatrix_p = dict(pmatrix)
	# create new initial state
	initial_states = {s: s_num
			for s, s_num in states_p.items()
			if eval(initial)}
	new_initial_state_num = max((states_p[s] for s in states_p)) + 1
	states_p[INITIAL_STATE] = new_initial_state_num
	for initial_state in initial_states:
		pmatrix_p[INITIAL_STATE, initial_state] = '1'
	# create new accepting state
	accepting_states = {s: s_num
			for s, s_num in states_p.items()
			if eval(accept)}
	new_accepting_state_num = new_initial_state_num + 1	
	states_p[ACCEPT_STATE] = new_accepting_state_num
	for accepting_state in accepting_states:
		pmatrix_p[accepting_state, ACCEPT_STATE] = '1'
	initial = 's == "{}"'.format(INITIAL_STATE)
	accept = 's == "{}"'.format(ACCEPT_STATE)	
	return (states_p, pmatrix_p), initial, accept

def get_pre_states(state, states, pmatrix, include_self_loops = True):
	if state not in states:
		return set()
	else:
		return set(s for (s, s_p) in pmatrix
			if s_p == state and (include_self_loops or s != s_p))
		
def get_post_states(state, states, pmatrix, include_self_loops = True):
	if state not in states:
		return set()
	else:
		return set(s_p for (s, s_p) in pmatrix
			if s == state and (include_self_loops or s != s_p))

def get_transitions(state, states, pmatrix, include_self_loops = True):
	if state not in states:
		return set()
	else:
		return set((s, s_p) for s, s_p in pmatrix
			if s_p == state and (include_self_loops or s != s_p))

def get_out_transitions(state, states, pmatrix, include_self_loops = True):
	if state not in states:
		return set()
	else:
		return set((s, s_p) for s, s_p in pmatrix
			if s == state and (include_self_loops or s != s_p))
	
#===============================================================================
#-------------------------------------------------------------------------------
# shell processes
#-------------------------------------------------------------------------------
#===============================================================================

def run_process(process_name, params):
	process = subprocess.Popen([process_name, *params], stdin = subprocess.PIPE,
			stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
	stdout = ''
	stdout, stderr = process.communicate()
	return stdout

def run_process_no_output(process_name, params):
	return subprocess.Popen([process_name, *params], stdin = subprocess.PIPE,
			stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

#===============================================================================
#-------------------------------------------------------------------------------
# elimination orders
#-------------------------------------------------------------------------------
#===============================================================================

def generate_elimination_order(D, random = False):
	states, pmatrix = D
	order = sorted(list(s for s, s_num in states.items()
			if s != INITIAL_STATE and s != ACCEPT_STATE),
			key = lambda x: states[x])
	if random:
		shuffle(order)
	return order

def compute_new_elimination_order(D, D_p, order):
	S, P = D
	S_p, P_p = D_p
	order_p = list(s for s in order if s in S_p)
	order_p.extend(list(s for s in S_p if s not in S))
	return order_p

#===============================================================================
#-------------------------------------------------------------------------------
# test stuff
#-------------------------------------------------------------------------------
#===============================================================================

def evaluate_final_expression(D):
	states, pmatrix = D
	print(eval(pmatrix[list(pmatrix)[0]]))

def evaluate_last_dag_node(G):
	vertices, edges = G
	print(eval(max(vertices, key = lambda x : len(x))))
	
def preprocess(D, initial, accept):
	states, pmatrix = D
	states, pmatrix = remove_p_zero_states(D, accept)
	states, pmatrix = make_accepting_states_absorbing((states, pmatrix), accept)
	(states, pmatrix), initial, accept = insert_init_and_accept((states, pmatrix),
			initial, accept)
	return (states, pmatrix), initial, accept 

def reconfigure(D_1, D_2, initial, accept, visualise = False, no_labels = False):
	# preprocessing for D_1
	D_1, initial_1, accept_1 = preprocess(D_1, initial, accept)
	# preprocessing for D_2
	D_2, initial_2, accept_2 = preprocess(D_2, initial, accept)

	states_1, pmatrix_1 = D_1
	states_2, pmatrix_2 = D_2

	infected = set(compute_infected(D_1, D_2)).difference({INITIAL_STATE,
		ACCEPT_STATE})
	infected = list(infected.intersection(set(states_1)))
	infected = sorted(infected, key = lambda s : states_1[s])
	order = generate_elimination_order(D_1, random = False)
	order = list(s for s in order if s not in infected)
	order.extend(infected)

	# build the dag for D_1
	D_1_d, G_1_d, M_1, num_products, num_sums, num_geos = build_dag(D_1, initial_1,
			accept_1, order, infected)
	#evaluate_final_expression(D_1_d)
	#evaluate_last_dag_node(G_1_d)
	
	# recalculate elimination order
	order = compute_new_elimination_order(D_1, D_2, order)
	# build the dag for D_2
	D_2_d, G_2_d, M_2, num_products, num_sums, num_geos = build_dag(D_2, initial_2,
			accept_2, order, infected)
	operations = num_products + num_sums + 2 * num_geos
	#evaluate_final_expression(D_2_d)
	#evaluate_last_dag_node(G_2_d)
	
	# rebuild dag
	D_2_d, G_2_d, M_2, num_products, num_sums, num_geos = rebuild_dag(D_1, D_2, M_1,
			initial_2, accept_2, order,
			visualise = visualise, display_reusability_results = False,
			no_labels = no_labels)
	reconfig_operations = num_products + num_sums + 2 * num_geos
	#evaluate_final_expression(D_2_d)
	#evaluate_last_dag_node(G_2_d)
	return operations, reconfig_operations

def create_predicates_for_synch_model(N, T):
	initial = 's == "{}"'.format(
			str(tuple(numpy.zeros(T, dtype = int))).replace(" ", ""))
	accept = ''
	for phase in range(T):
		zeros = numpy.zeros(T, dtype = int)
		zeros[phase] = N
		tuple_str = 's == "{}"'.format(str(tuple(zeros)).replace(" ", ""))
		accept += tuple_str
		if phase < T - 1:
			accept += ' or '
	return initial, accept	

def generate_zeroconf(n):
	states = dict()
	pmatrix = dict()
	states['err'] = -1
	states['ok'] = 0
	states['1'] = 1
	states['2'] = 2
	pmatrix['1', 'ok'] = '(1-q)'
	pmatrix['1', '2'] = 'q'
	pmatrix['2', '1'] = '(1-p)'
	pmatrix['2', 'err'] = 'p'
	last_state = '2'
	for i in range(n):
		next_state = str(eval(last_state) + 1)
		states[next_state] = eval(next_state) + 1
		del pmatrix['1', last_state]
		pmatrix['1', next_state] = 'q'
		pmatrix[next_state, last_state] = 'p'
		pmatrix[next_state, '1'] = '(1-p)' 
		last_state = next_state
	return states, pmatrix


def generate_synch(N, T, epsilon, R):
	run_process(SYNCH_MODEL_GENERATOR,
			['{}'.format(N), '{}'.format(T), '{}'.format(epsilon),
					'{}'.format(R)])
	D = get_automaton_from_files(
			'{}{}_{}_{}_{}.sta'.format(MODEL_PATH, N, T, epsilon, R),
			'{}{}_{}_{}_{}.tra'.format(MODEL_PATH, N, T, epsilon, R))
	run_process('rm', [
			'{}{}_{}_{}_{}.sta'.format(MODEL_PATH, N, T, epsilon, R),
			'{}{}_{}_{}_{}.tra'.format(MODEL_PATH, N, T, epsilon, R)])
	return D

def main():
#===============================================================================
# zeroconf
#===============================================================================	
	'''
	total_operations = 0
	total_reconfig_operations = 0
	print('n, op, reop, perc')
	for N in range(1, 201):
		D_1 = generate_zeroconf(N)
		D_2 = generate_zeroconf(N + 1)
		initial = 's == "1"'
		accept = 's == "err"'
		operations, reconfig_operations = reconfigure(D_1, D_2, initial,
				accept, visualise = False)
		total_operations += operations
		total_reconfig_operations += reconfig_operations
		percentage = (total_reconfig_operations / total_operations * 100)
		print('{},{},{},{},'.format(N, total_operations,
				total_reconfig_operations, percentage))
	'''

#===============================================================================
# synch
#===============================================================================	
	
	print('T, R, op, reop, perc')
	for N in range(5, 6): #range(4, 6):
		for T in range(4, 8): #range(3, 7):
			total_operations = 0
			total_reconfig_operations = 0
			for R in range(0, T):#range(1, T - 1):
				epsilon = 0.1
				D_1 = generate_synch(N, T, epsilon, R)
				D_2 = generate_synch(N, T, epsilon, R + 1)
				initial, accept = create_predicates_for_synch_model(N, T)
				operations, reconfig_operations = reconfigure(D_1, D_2,
						initial, accept, visualise = False)
				total_operations += operations
				total_reconfig_operations += reconfig_operations
				percentage = (total_reconfig_operations / total_operations * 100)
				print('{},{},{},{},{},'.format(T, R, total_operations,
						total_reconfig_operations, percentage))
	
	
	
if __name__ == '__main__':
	main()


