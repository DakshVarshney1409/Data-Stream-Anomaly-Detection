import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def generate_data_stream(period=24, season_period=100, noise_level=0.5, step=1):
    """
    Generates a continuous stream of data with regular patterns, seasonal changes, and random noise.

    :param period: Period for regular patterns (e.g., 24 for a daily cycle).
    :param season_period: Number of steps after which seasonal patterns change.
    :param noise_level: The level of random noise to add.
    :param step: The step size for each iteration (default is 1).
    :return: Yields a new data point in each iteration.
    """
    timestep = 0
    while True:
        # Regular pattern (e.g., daily cycle)
        regular_pattern = np.sin(2 * np.pi * timestep / period)

        # Seasonal variation
        seasonal_variation = np.sin(2 * np.pi * timestep / season_period)

        # Random noise
        noise = noise_level * np.random.randn()

        # Combined signal
        data_point = regular_pattern + seasonal_variation + noise

        yield data_point

        timestep += step
        time.sleep(0.1)  # Simulate real-time data streaming with a slight delay


# generate data stream finished 
# Assuming generate_data_stream() is the function we defined earlier

# Step 1: Setup Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

# Step 2: Preprocess and Train the Model
# For simplicity, we'll train the model on a small batch of simulated data
train_data = [next(generate_data_stream()) for _ in range(1000)]
train_data = np.array(train_data).reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# Train the model
model.fit(train_data_scaled)

# Step 3: Anomaly Detection in Real-time Stream
def detect_anomalies(data_stream, model, scaler):
    for data_point in data_stream:
        # Scale the data point
        scaled_data = scaler.transform([[data_point]])

        # Predict anomaly (1 for normal, -1 for anomaly)
        prediction = model.predict(scaled_data)
        if prediction == -1:
            print(f"Anomaly detected: {data_point}")

# finished the function for anomaly detection

# Assuming the existing setup for Isolation Forest and data stream generation

# Setup Matplotlib plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'b-', animated=True)  # Normal data points
anom, = plt.plot([], [], 'ro', markersize=8, animated=True)  # Anomalies
anomaly_points = []
# Initialize the plot
def init():
    ax.set_xlim(0, 50)  # Adjust based on your expected range
    ax.set_ylim(-3, 3)  # Adjust based on your expected range
    return ln, anom

# Update function for animation
def update(frame):
    xdata.append(frame)
    ydata.append(next(generate_data_stream()))

    # Detect anomaly
    scaled_data = scaler.transform([[ydata[-1]]])
    prediction = model.predict(scaled_data)
    
    ln.set_data(xdata, ydata)
    if prediction == -1:
        print(f"Anomaly detected: {ydata[-1]}")
        anomaly_points.append((xdata[-1], ydata[-1]))

    # Plot anomalies
    if len(anomaly_points) > 0:
        x_anom, y_anom = zip(*anomaly_points)
        anom.set_data(x_anom, y_anom)
        
    if len(xdata) > 50:
        ax.set_xlim(xdata[-50], xdata[-1])

    return ln, anom

#detect_anomalies(generate_data_stream(), model, scaler)

ani = FuncAnimation(fig, update, init_func=init,cache_frame_data=False, blit=True, interval=100, frames=200)
#plt.show()
writergif = animation.PillowWriter(fps=30)
ani.save('anomaly.gif', writer = writergif) 
plt.grid()
plt.show(block=False)
plt.pause(6)
plt.close()