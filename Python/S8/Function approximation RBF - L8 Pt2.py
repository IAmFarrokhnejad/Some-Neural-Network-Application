#Purpose of the work: Design and implementation of an RBF network with Python to estimate body fat percentage based on various body measurements provided in the bodyfat dataset
#About the dataset(bodyfat): Each input feature is a 252x1 vector; Stored attributes: age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist


#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner



import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Function to calculate radial basis values for a set of inputs
def rbf(x, centers, spread):
    res = np.zeros((x.shape[0], len(centers)))
    try:
        for j in range(x.shape[1]):
            for i, c in enumerate(centers):
                res[:, i] += (np.exp(-(x[:,j].T - c)**2 / (2 * (spread**2)))).T
    except IndexError:    
        for i, c in enumerate(centers):
            res[:, i] += (np.exp(-(x - c)**2 / (2 * (spread**2)))).T
    finally:
        return res

# Assuming bodyfat.csv contains bodyfatInputs and bodyfatTargets
data = np.genfromtxt('C:/Users/afr51/Downloads/Telegram Desktop/bodyfat.csv', delimiter=',')
bodyfatInputs = data[:-1, :].T # Each row contains one data sample
bodyfatTargets = data[-1, :].reshape(-1,1) # Column vector of classes (each row corresponds to respective row in X) 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bodyfatInputs, bodyfatTargets, test_size=0.2, random_state=3)

# Standardize input data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Start training process
centers_list = [150]
for n_centers in centers_list:

    # Perform k-means clustering to choose centers        
    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(X_train.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    # Pass inputs through hidden layer and standardize
    Xtr_rbf = rbf(X_train_std, centers, 1)
    Xtst_rbf = rbf(X_test_std, centers, 1)
    Xtr_rbf_std = scaler.fit_transform(Xtr_rbf)
    Xtst_rbf_std = scaler.transform(Xtst_rbf)

    # Create SLP for the rest of the network
    net = nl.net.newp(np.column_stack([Xtr_rbf_std.min(axis=0), Xtr_rbf_std.max(axis=0)]), 1, transf= nl.net.trans.PureLin())

    # Train network
    net.train(Xtr_rbf_std, y_train.reshape(-1, 1), epochs=1000, show=10, goal=1e-7, lr=0.00028)

    # Get outputs
    ytr_pred = net.sim(Xtr_rbf_std)
    yts_pred = net.sim(Xtst_rbf_std)

    # Calculate and print the Mean Squared Error (MSE) on the training set
    mse_train = mean_squared_error(y_train, ytr_pred)
    mse_test = mean_squared_error(y_test, yts_pred)
    print(f'NEWRB, neurons = {centers}, MSE on training set = {mse_train}, MSE on testing set = {mse_test}')

# Predict the output on the entire dataset
Y = net.sim(np.vstack([Xtr_rbf_std, Xtst_rbf_std]))

# Plot regression
plt.figure()
plt.plot(bodyfatTargets, Y, 'o')
plt.title('Regression plot')
plt.xlabel('Actual Body Fat')
plt.ylabel('Predicted Body Fat')
plt.show()
