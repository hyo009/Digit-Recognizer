import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
#matplotlib inline

# load data
labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# make image binary
test_images[test_images>0]=1
train_images[train_images>0]=1
# view an image
i=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
#plt.show()
plt.title(train_labels.iloc[i,0])
# plot histogram
plt.hist(train_images.iloc[i])
#plt.show()
# try svm.SVC
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

# label the test data
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])

df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
