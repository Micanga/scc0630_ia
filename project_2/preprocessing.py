import numpy as np
import operator

def enrolments():
	# 1. Defining important header index semantically
	ID, UNENROLLED = 0, 2

	# 2. Extracting the informations of interest
	# - Enrolled Learners and Unenrolled Learners
	learners = list()
	n_enrolled, n_uenrolled = 0, 0
	with open('dataset/big-data-2_enrolments.csv') as enrolments:
		for learner_info in enrolments:
			info = learner_info.split(',')
			if info[UNENROLLED] == "": # learner quit? No, so quit == False
				n_enrolled += 1
				learners.append([info[ID],False])
			else: # learner quit? Yes, so quit == True
				n_uenrolled += 1
				learners.append([info[ID],True])

	# 3. Returning the preprocessed information
	print('ENROLLED INFORMATION EXTRACTED FEATURES')
	print('- Enrolled learners =',n_enrolled)
	print('- Unenrolled learners =',n_uenrolled)
	print('- Unenroll rate =',float(n_uenrolled)/n_enrolled)
	learners = sorted(learners,key=lambda x: (x[1], x[0]))
	learners.remove(['learner_id', True])

	return learners

def questions():
	# 1. Defining important header index semantically
	ID, QUIZ_QUESTION, WEEK_NUMBER, STEP_NUMBER,\
	 QUESTION_NUMBER, CORRECT = 0, 1, 2, 3, 4, 7

	# 2. Extracting the information of interest
	# - Submissions info
	# - Question set
	submit_info = list()
	question_set = set()
	with open('dataset/big-data-2_question-response.csv') as questions:
		for questions_info in questions:
			info = questions_info.split(',')
			question_set.add((info[QUIZ_QUESTION],info[QUESTION_NUMBER]))
			submit_info.append([info[ID],info[WEEK_NUMBER],\
				info[QUIZ_QUESTION],info[QUESTION_NUMBER],\
				1 if info[CORRECT][:-1] == 'true' else 0])

	print('QUESTION SET EXTRACTED')
	question_set = sorted(question_set,key=lambda x: (x[0],x[1]))
	question_set.remove(('quiz_question', 'question_number'))
	print('- Number of questions =',len(question_set))

	# 3. Returning the preprocessed information
	print('SUBMIT INFORMATION EXTRACTED')
	submit_info = sorted(submit_info,key=lambda x: \
		(x[1], x[2], x[3], x[0], x[4]))
	submit_info.remove(['learner_id', 'week_number', \
		'quiz_question', 'question_number', 0])
	print('- Number of sumissions =',len(submit_info))

	return submit_info, question_set

def split_learners(enrolments,questions):
	# 1. Initializing the dictionaries
	noquit_learners,quit_learners = dict(), dict()

	# 2. Identifying the quit and no quit learners
	for learner in enrolments:
		if learner[1] == True:
			quit_learners[learner[0]] = list()
		else:
			noquit_learners[learner[0]] = list()

	# 3. Relating the submited questions with each
	# enrolled learner
	for submit in questions:
		if submit[0] in quit_learners:
			quit_learners[submit[0]].append(submit[1:])
		else:
			noquit_learners[submit[0]].append(submit[1:])

	# 4. Returning the splitted dataset and concatenated
	return noquit_learners, quit_learners

def transform_nn_input(noquit_learners,quit_learners,question_set):
	# 0. Defining the variables
	dataset = dict()
	n = len(question_set)

	# 1. Adding the no quit learners to the dataset
	for learner in noquit_learners:
		# a. creating the activity matrix and labeling it
		# 0 = no quit learners
		dataset[learner] = [-1*np.ones(shape=(n,n)),np.zeros(n)]

		# b. extracting the submit questions by the learner
		noquit_learner_questions = [noquit_learners[learner][i][1] \
			for i in range(len(noquit_learners[learner]))]

		# c. formating the matrix
		for question_number in range(len(question_set)):
			if question_set[question_number][0] in noquit_learner_questions:
				question_index = noquit_learner_questions.index(question_set[question_number][0])
				for i in range(question_number,n):
					dataset[learner][0][i][question_number] = noquit_learners[learner][question_index][3]

	print('- No quit Learners =',len(dataset))

	# 2. Adding the quit learners to the dataset
	for learner in quit_learners:
		# a. creating the activity matrix and labeling it
		# 1 = quit learners
		dataset[learner] = [-1*np.ones(shape=(n,n)),np.ones(n)]

		# b. extracting the submit questions by the learner
		quit_learner_questions = [quit_learners[learner][i][1] \
			for i in range(len(quit_learners[learner]))]

		# c. formating the matrix
		for question_number in range(len(question_set)):
			if question_set[question_number][0] in quit_learner_questions:
				question_index = quit_learner_questions.index(question_set[question_number][0])
				for i in range(question_number,n):
					dataset[learner][0][i][question_number] = quit_learners[learner][question_index][3]

	print('- Total Learners =',len(dataset))

	return dataset