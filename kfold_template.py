#this is the template we use when we want to separate and test models 
#against different sets of data

import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics



def run_kfold(data, target, split_number, machine, use_accuracy=0, use_confusion=0):
	print("run kfold")
	kfold_object = KFold(n_splits=split_number)
	kfold_object.get_n_splits(data)

	results_r2 = []
	results_accuracy = []
	results_confusion = []

	for training_index, test_index in kfold_object.split(data):
		data_training = data[training_index]
		target_training = target[training_index]
		data_test = data[test_index]
		target_test = target[test_index]
		#machine = linear_model.LinearRegression()
		machine.fit(data_training, target_training)
		print(machine.score(data_training, target_training))
		new_target = machine.predict(data_test)
		results_r2.append(metrics.r2_score(target_test, new_target))
		if use_accuracy == 1:
			results_accuracy.append(metrics.accuracy_score(target_test, new_target))
		if use_confusion == 1:
			results_confusion.append(metrics.confusion_matrix(target_test, new_target))
	return results_r2, results_accuracy, results_confusion




if __name__ == '__main__':
	dataset = pandas.read_csv("dataset.csv")
	target = dataset.iloc[:,0].values
	data = dataset.iloc[:,3:9].values
	run_kfold(data, target, 4)