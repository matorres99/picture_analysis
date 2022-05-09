import pandas
import os
import glob

import imageio

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import kfold_template

category_dataset = pandas.read_csv("pictures_category.csv")
# print(category_dataset)



def getrgb(file_path):
	imimage = imageio.imread(file_path, pilmode="RGB")
	imimage = imimage/255
	imimage = imimage.sum(axis = 0).sum(axis = 0)/(imimage.shape[0]*imimage.shape[1])
	return imimage

# print(getrgb("pictures/pic01.jpeg"))


def read_picture_folder(folder_name):
	result = pandas.DataFrame()
	for file_path in glob.glob(folder_name + "/*"):
		image_features = pandas.DataFrame(getrgb(file_path))
		image_features = pandas.DataFrame.transpose(image_features)
		image_features["filename"] = file_path.replace(folder_name + "/","")
		result = pandas.concat([result, image_features])
	result = result.rename(columns={0: "red", 1:"green", 2:"blue"})
	return result	

image_dataset = read_picture_folder("pictures")

# print(image_dataset)

dataset = pandas.merge(image_dataset, category_dataset, on="filename")

print(dataset)


picture_attributes_count = len(image_dataset.columns) - 1

data = dataset.iloc[:,0:picture_attributes_count].values

target = dataset.iloc[:,(picture_attributes_count+1)].factorize()
target_index = target[1]
target = target[0]
# print(target)


# machine = linear_model.LogisticRegression()
machine = RandomForestClassifier(criterion="gini", max_depth=30, n_estimators = 200, bootstrap = True, max_features="auto")
results = kfold_template.run_kfold(data, target, 3, machine, 1, 1)

print(results[1])
for i in results[2]:
	print(i)

machine = RandomForestClassifier(criterion="gini", max_depth=30, n_estimators = 200, bootstrap = True, max_features="auto")
machine.fit(data, target)


new_pictures_dataset = read_picture_folder("new_pictures")

prediction = machine.predict(new_pictures_dataset.iloc[:,0:picture_attributes_count])
prediction = list(target_index[prediction])
new_pictures_dataset['prediction'] =prediction
print(new_pictures_dataset)

