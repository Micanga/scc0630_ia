# method that calculates the entropy of a data set in 
# a decision tree.
# - tuple: tuple with the occurrence number for each 
# class;
# - m: number of classes.
info <- function(tuple,m){
	result = 0
	for(i in 1:m){
		if(tuple[i] != 0){
			p = tuple[i]/sum(tuple)
			result = result - (p)*log2(p)
		}
	}
	return(result)
}

# method that calculates the entropy of a data set based 
# on a partition for a decision tree.
# - tuples: vector of tuples (matrix(n,m)) with the occurrence
# number for each class;
# - n: number of tuples;
# - m: number of classes;
# - total: total number of data = |D|.
infoA <- function(tuples,n,m,total){
	result = 0
	for(i in 1:n){
		result = result + (sum(tuples[i,])/total)*info(tuples[i,],m)
	}
	return(result)
}

# method that calculates the information gain for a decision tree
# info_d: data set entropy
# info_a: partition set entropy
gain <- function(info_d,info_a){
	return(info_d-info_a)
}