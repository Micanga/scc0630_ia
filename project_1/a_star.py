from State import State
from copy import copy

class SearchAStar:

	#  initialisation method
	def __init__(self, starte_state, actions):
		# structural parameters
		self.root = NodeAStar(starte_state, 0, None)

		# a-star parameters
		self.actions = actions
		self.activeProcesses = list()
		self.activeProcesses.append(self.root)

	# a-star search method
	def search(self, termine_state):
		# a. initialising the visit counter
		visit_counter = 0

		# b. starting the a-star search
		while len(self.activeProcesses) > 0:
			#print("Evaluation:")
			#print(self.activeProcesses[0].evaluation)
			#print("Status")
			#print(self.activeProcesses[0].state.status)

			# i. activating the process for the current
			# choosen node
			self.activateProcesses()

			# ii. removing the current node from the
			# activated processes list
			self.activeProcesses.pop(0)

			# iii. counting the visit, evaluating the
			# new activated processes and ranking it
			visit_counter += 1
			if len(self.activeProcesses) > 0: 
				self.evaluateActiveProcesses() # calculating the heuristic function value
				self.rankEvaluation() # ranking the activated nodes

				# iv. checking the stop condition
				if self.activeProcesses[0].state.is_equal(termine_state):
					#print 'Solution found'
					#print '- Visited Nodes: '+str(visit_counter)
					#self.show_route(self.activeProcesses[0])
					return activeProcesses[0]
			else:
				print 'No solution'
				return None

	# processes activation method
	def activateProcesses(self):
		# a. taking the current process/node
		cur_node = self.activeProcesses[0]

		# b. expanding the node
		for a in self.actions:
			new_state = cur_node.state.move(a)
			if new_state is not None:
				cur_node.addChild(new_state)

		# c. adding its child to the active
		# processes list
		for c in cur_node.childs:
			self.activeProcesses.append(c)

	# heuristic calculation method
	def evaluateActiveProcesses(self):
		for node in self.activeProcesses:
			# a. reseting the heuristic value
			node.heuristic = 0

			# b. calculating the number of restrictions
			# for the given state
			for a in self.actions:
				new_state = node.state.move(a)
				if new_state is not None:
					node.heuristic += 1

			# c. calculating the heuristic function value
			#	f(x) = Manhattan(x) + Restrictions(x) + Depth(x) 
			node.evaluation = (6-node.state.status[0]-node.state.status[1])+\
			(node.heuristic) + node.depth

	# active processes ranking method
	def rankEvaluation(self):
		self.activeProcesses.sort(reverse=False,key=lambda x: x.evaluation)

	# print the route to the node into
	# the tree method
	def show_route(self,node):
		# a. initialising the route vector
		# and the current node
		route = []
		cur_node = copy(node)

		# b. building the route vector by
		# node's father backtracking
		while cur_node is not None:
			route.append(cur_node.state.status)
			cur_node = copy(cur_node.father)

		# c. showing the result
		print '- Steps:', len(route)-1,'\n- Route:', list(reversed(route))


class NodeAStar(State):
	def __init__(self, state=None, depth=0, father=None):
		# Estado pertencente ao nó
		self.state = state

		# Estrutura da Arvore de caminhos A*
		self.father = father
		self.childs = []

		# Função avaliação
		self.depth = depth  # Custo do caminho = profundidade

		# Heuristica => Quanto menos pessoas na margem de origem mais proximo do objetivo
		self.heuristic = 0  # Heuristic = Total de pessoas na margem origem
		self.evaluation = 0  # Função avaliadora >>> f(n) = g(n) + h(n)

	def addChild(self, state):
		# Criando nó filho
		child = NodeAStar(state, self.depth + 1, self)

		# Add compoor em filhos do pai
		self.childs.append(child)