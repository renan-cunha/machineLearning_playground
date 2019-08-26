# Linear Regression With Multiple Variables

Uses batch gradient descent

```python
dataset = load_boston()

#  getting average number of rooms per dwelling and the property-yax per
#  10.000
x = dataset.data[:, [5, 9]]
y = dataset.target
# preprocessing
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.reshape(y.shape[0], 1))
y = y.flatten()

model = LinearRegressor()
errors = model.fit(x, y, num_iters=10000, learning_rate=0.001)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
axes = Axes3D(fig)
axes.scatter(xs=x[:, 0], ys=x[:, 1], zs=y)
    
length = 10000
x = np.linspace(0, 1, num=length)
data = np.empty((length, 2))
data[:, 0] = x
data[:, 1] = x
result = model.predict(data) 

axes.plot(xs=x, ys=x, zs=result.flatten())
axes.set_xlabel("Number of Rooms")
axes.set_ylabel("Tax Value")
axes.set_zlabel("Price")
plt.show()

plt.plot(errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()
```
