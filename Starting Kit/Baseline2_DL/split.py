import pandas as pd 
import random

data = pd.read_csv("newdata.tsv", sep = '\t', header = None)
# train = []
# test = []

# choices = []
# i = 0
# while i < data.shape[0]:
# 	choices.append(i)
# 	i+=10

# while len(train) < 0.9*data.shape[0]:
# 	start = random.choice(choices)
# 	for i in range(10):
# 		x = data.iloc[start + i].tolist()
# 		train.append(x)
# 	choices.remove(start)

# for start in choices:
# 	for i in range(10):
# 		x = data.iloc[start + i].tolist()
# 		test.append(x)

# trainDataFrame = pd.DataFrame(train)
# validationDataFrame = pd.DataFrame(test)

trainDataFrame = data[0:int(len(data)/4*0.9)*4]
validationDataFrame = data[int(len(data)/4*0.9)*4:]

trainDataFrame.to_csv("traindata.tsv", sep='\t') 
validationDataFrame.to_csv("validationdata.tsv", sep='\t')
