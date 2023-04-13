import pandas as pandas
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier 
from scipy.optimize import minimize 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import plotly.graph_objs as go
import plotly
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
################Cleaning up data####################

print("*******************Cleaning Up Data*******************")
print("")
print("")
print("")
n = 6 #number of features

data = np.loadtxt("Data.txt", dtype=float)
feature_matrix = np.zeros([len(data),n-1])
target_var = np.zeros(len(data))

for i in range(len(data)):
    for j in range(n-1):
        feature_matrix[i,j] = data[i,j]
        target_var[i] = data[i,-1]

target_var = target_var[:,np.newaxis]

##############Creating the model######################

print("*********************************************************")
print("")
print("")
print("")

model0 = LinearRegression(fit_intercept=True)

print("*******************Creating the Fit with Model 1*******************")
print("")
print("")
print("")

model0.fit(feature_matrix,target_var)
coefficients = model0.coef_
intercept = model0


print("")
print("")
print("")

#####################Training and Testing with Score#####

print("*******************Creating Training set with Model 2*******************")
print("")
print("")
print("")

Xtrain, Xtest, ytrain, ytest = train_test_split(feature_matrix, target_var, random_state=1)

ytrain = ytrain.astype('int')
ytest = ytest.astype('int')
model1 = GaussianNB()

model1.fit(Xtrain, ytrain)
y_model1 = model1.predict(Xtest)


#print(accuracy_score(ytest, y_model1))

##################Cross Validation#######################


X1model2, X2model2, y1model2, y2model2 = train_test_split(feature_matrix, target_var, random_state=0,train_size=0.5)

model2 = KNeighborsClassifier(n_neighbors=1)

y1model2 = y1model2.astype('int')
X1model2 = X1model2.astype('int')

model2.fit(X1model2, y1model2) # evaluate the model on the second set of data
y2model2_model2 = model2.predict(X2model2)
#print(accuracy_score(y2model2, y2model2_model2))


################Visualzing data#######################


data_new = pd.read_excel('Data.xlsx')
#print(data_new)

markersize = data_new['Salt Concentration(mM)']/12
markercolor = data_new['Time(hours)']
markershape = data_new['Temperature(Celcius)'].replace("four","square").replace("two","circle")

############Testing different values####################
print("#####################Computing Predictions##################################")
print("")
print("")
print("")

test_features = [[[0.1, 10, 60,0.25, 0.1]],
                 [[0.05, 10, 60,0.5, 0.1]],
                 [[0.01, 10, 60,0.75, 0.1]],
                 [[0.05, 10, 60,1, 0.1]],
                 [[0.05, 14, 60,1.5, 0.1]],
                 [[0.1, 10, 60,0.5, 0.1]],
                 [[0.01, 14, 60,0.75, 0.1]],
                 [[0.1, 14, 60,0.05, 0.1]],
                 [[1, 14, 60,0.25, 0.1]],
                 [[0.2, 14, 60,0.25, 0.1]],
                 [[0.3, 14, 60,0.25, 0.1]],
                 [[0.4, 14, 60,0.25, 0.1]],
                 [[0.5, 14, 60,0.25, 0.1]],
                 [[0.1, 20, 60,0.1, 0.1]]]


print("Using First Model")
print("")
print("")
print("")
print("Data presented in [Salt concentration(mM), Time(hours), Temperature(C), Dye Conc(mM), Ratio, Size(nm)]")
print("")
print("")
print("")
for i in test_features:
    
    size_prediction = model0.predict(i)
    print("Size Prediction for", i, " = ", size_prediction)
    print("")

print("")
print("")
print("Using Second Model")
print("")
print("")
print("")

for i in test_features:
    size_prediction = model1.predict(i)
    print("Size Prediction for", i, " = ", size_prediction)
    print("")



print("")
print("")
print("Using Third Model")
print("")
print("")
print("")

for i in test_features:
    size_prediction = model2.predict(i)
    print("Size Prediction for", i, " = ", size_prediction)
    print("")


##########CREATE A SURFACE PLOT FOR DIFFERENT PARAMETERS

KClconc = feature_matrix[:,0]
time = feature_matrix[:,1]
dyeconc = feature_matrix[:,3]
particlesize = target_var.copy()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(KClconc, time)
Z = particlesize



# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


