from copy import copy
import time

class Node(object):

	# node intialisation method
	def __init__(self, state, depth = 0, father = None):
		# node strutuctural parameters
		self.depth = depth
		self.father = father
		self.childs = []

		# node evaluation parameters
		self.state = state

	# add child method
	def add_child(self,child):
		# a. updating child parameters
		child.father = self
		child.depth = self.depth + 1

		# b. appending the child
		self.childs.append(child)

		return child

	# add child method by state
	def add_child_state(self,state):
		# a. creating a child node
		new_child = Node(state,self.depth+1,self)

		# b. appending the child
		self.childs.append(new_child)

		return new_child

class SearchTree(object):
	
	# search tree initialisation method
	def __init__(self,root = None,max_depth = 10):
		# tree structural parameters
		self.root = root
		self.max_depth = max_depth

	# update the depths from a node method
	def update_depths(self,node,depth):
		if node is None:
			return

		node.depth = depth
		for child in node.childs:
			self.update_depths(child,depth+1)

	# depth-first search algorithm (initialisation)
	def dfs(self,node,state):
		# a. initialising the visited nodes
		# counter for dfs
		self.dfs_counter = 0

		# b. starting the search
		start_time = time.time()
		result = self.recursive_dfs(node,state)
		end_time = time.time()
		time.sleep(1)

		# c. printing/saving the result
		if result is not None:
			#print '- Visited Nodes:', result[1]
			self.show_route(result[0])
			#print '- Execution Time:', (end_time - start_time)

			file = open("outputDFS"+str(self.max_depth)+".csv","a")
			file.write('- Visited Nodes: '+str(result[1])+"\n")
			file.write('- Execution Time: '+str(end_time - start_time)+"\n")
			file.close()

		else:
			print ('- No solution found.')

	# depth-first search algorithm (recursive search)
	def recursive_dfs(self,node,state):
		# a. checking the stop condition
		if node.state.is_equal(state):
			result = [node, self.dfs_counter]
			return result

		# b. searching
		self.dfs_counter += 1
		for child in node.childs:
			result = self.recursive_dfs(child,state)
			if result is not None:
				return result

	# breadth-first search algorithm (initialisation)
	def bfs(self,node,state):
		# a. initialising the bfs queue and
		# the visited nodes counter for bfs
		self.bfs_queue = list()
		self.bfs_counter = 0

		# b. starting the search
		start_time = time.time()
		result = self.recursive_bfs(node,state)
		end_time = time.time()
		time.sleep(1)

		# c. printing the result
		if result is not None:
			#print '- Visited Nodes:', result[1]
			#self.show_route(result[0])
			#print '- Execution Time:', (end_time-start_time)

			file = open("outputBFS"+str(self.max_depth)+".csv","a")
			file.write('- Visited Nodes: '+str(result[1])+"\n")
			file.write('- Execution Time: '+str(end_time - start_time)+"\n")
			file.close()

		else:
			print ('- No solution found.')

	# breadth-first search algorithm (recursive search)
	def recursive_bfs(self,node,state):
		# a. checking the stop condition
		if node.state.is_equal(state):
			return node, self.bfs_counter

		# b. removing the current node from the queue
		elif len(self.bfs_queue) != 0:
			self.bfs_queue.pop(0)

		# c. adding the childs to the queue
		for child in node.childs:
			self.bfs_queue.append(child)

		# d. searching
		self.bfs_counter += 1
		if len(self.bfs_queue) != 0:
			result = self.recursive_bfs(self.bfs_queue[0],state)
			return result
		else:
			return None

	# print tree method
	def show(self,node):
		print (node.depth,node.state.status)
		for child in node.childs:
			self.show(child)

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
		print( '- Steps:', len(route)-1,'\n- Route:', list(reversed(route)))