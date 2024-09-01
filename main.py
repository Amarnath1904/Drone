import kivy
kivy.require('1.11.1')  # Replace with your version

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock

import time
import math
import numpy as np
from simple_pid import PID
from plyer import accelerometer, gyroscope, gps







class DroneControlApp(BoxLayout):
    def __init__(self, **kwargs):
        super(DroneControlApp, self).__init__(**kwargs)
        self.pitch = 0.0
        self.roll = 0.0
        self.previous_time = time.time()
        init_sensors()
        init_gps()  # Initialize the GPS
        Clock.schedule_interval(self.update, 1.0 / 50.0)  # Run at 50 Hz

    def update(self, dt):
        current_time = time.time()
        delta_time = current_time - self.previous_time
        self.previous_time = current_time

        # Read sensor data
        gyro_x, gyro_y, gyro_z = read_gyroscope()
        acc_x, acc_y, acc_z = read_accelerometer()
        latitude, longitude, altitude = get_gps_data()  # Get the latest GPS data

        # Perform sensor fusion to get pitch and roll
        self.pitch, self.roll = sensor_fusion(acc_x, acc_y, acc_z, gyro_x, gyro_y, delta_time, self.pitch, self.roll)

        # Calculate PID outputs for pitch and roll
        pitch_output = pid_pitch(self.pitch)
        roll_output = pid_roll(self.roll)

        # Update labels with the new data
        self.ids.pitch_label.text = f"Pitch: {self.pitch:.2f}"
        self.ids.roll_label.text = f"Roll: {self.roll:.2f}"
        self.ids.gps_label.text = f"GPS: Lat {latitude:.4f}, Lon {longitude:.4f}, Alt {altitude:.2f}"
        self.ids.pitch_output_label.text = f"Pitch Output (PID): {pitch_output:.2f}"
        self.ids.roll_output_label.text = f"Roll Output (PID): {roll_output:.2f}"

        # Optionally print GPS data for debugging
        print(f"GPS Data: Lat {latitude}, Lon {longitude}, Alt {altitude}")










# Kalman Filter Class
class KalmanFilter:
    def __init__(self, Q_angle=0.001, Q_bias=0.003, R_measure=0.03):
        self.Q_angle = Q_angle
        self.Q_bias = Q_bias
        self.R_measure = R_measure

        self.angle = 0.0  # Reset the angle
        self.bias = 0.0  # Reset the bias
        self.rate = 0.0  # Reset the rate

        self.P = np.zeros((2, 2))  # Error covariance matrix
        self.K = np.zeros(2)  # Kalman gain

    def update(self, new_angle, new_rate, dt):
        # Predict phase
        self.rate = new_rate - self.bias
        self.angle += dt * self.rate

        # Update error covariance matrix
        self.P[0][0] += dt * (dt*self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt

        # Compute Kalman gain
        S = self.P[0][0] + self.R_measure
        self.K[0] = self.P[0][0] / S
        self.K[1] = self.P[1][0] / S

        # Update estimate with measurement zk (new_angle)
        y = new_angle - self.angle
        self.angle += self.K[0] * y
        self.bias += self.K[1] * y

        # Update the error covariance matrix
        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]

        self.P[0][0] -= self.K[0] * P00_temp
        self.P[0][1] -= self.K[0] * P01_temp
        self.P[1][0] -= self.K[1] * P00_temp
        self.P[1][1] -= self.K[1] * P01_temp

        return self.angle, self.bias

# PID setup
pid_pitch = PID(1.0, 0.1, 0.05, setpoint=0)
pid_roll = PID(1.0, 0.1, 0.05, setpoint=0)

# Set PID output limits (adjust based on motor control range)
pid_pitch.output_limits = (-500, 500)
pid_roll.output_limits = (-500, 500)

# Kalman filter setup for pitch and roll
kalmanFilterPitch = KalmanFilter()
kalmanFilterRoll = KalmanFilter()

# Global variables to store the latest GPS coordinates
current_latitude = 0.0
current_longitude = 0.0
current_altitude = 0.0

# Callback function to update GPS coordinates
def on_location(**kwargs):
    global current_latitude, current_longitude, current_altitude
    current_latitude = kwargs.get('lat', 0.0)
    current_longitude = kwargs.get('lon', 0.0)
    current_altitude = kwargs.get('altitude', 0.0)
    print(f"GPS Updated: Lat {current_latitude}, Lon {current_longitude}, Alt {current_altitude}")

# Initialize and start GPS
def init_gps():
    try:
        gps.configure(on_location=on_location, on_status=on_status)
        gps.start(minTime=1000, minDistance=0)
    except NotImplementedError:
        print("GPS is not supported on this device")

# Handle GPS status updates (optional, for debugging)
def on_status(stype, status):
    print(f"GPS Status: {stype}, {status}")

# Function to get the latest GPS data
def get_gps_data():
    global current_latitude, current_longitude, current_altitude
    return current_latitude, current_longitude, current_altitude

# Initialize sensors
def init_sensors():
    try:
        accelerometer.enable()
        gyroscope.enable()
    except NotImplementedError:
        print("Accelerometer or Gyroscope not supported on this device")

# Read gyroscope data using Plyer
def read_gyroscope():
    gyro_data = gyroscope.rotation
    if gyro_data:
        return gyro_data[0], gyro_data[1], gyro_data[2]  # (gyro_x, gyro_y, gyro_z)
    else:
        return 0.0, 0.0, 0.0  # Default values if no data

# Read accelerometer data using Plyer
def read_accelerometer():
    acc_data = accelerometer.acceleration
    if acc_data:
        return acc_data[0], acc_data[1], acc_data[2]  # (acc_x, acc_y, acc_z)
    else:
        return 0.0, 0.0, 1.0  # Default values if no data

# Sensor fusion using Kalman filter
def sensor_fusion(acc_x, acc_y, acc_z, gyro_x, gyro_y, dt, pitch, roll):
    # Calculate pitch and roll from the accelerometer
    pitch_acc = math.atan2(acc_y, acc_z) * 180 / math.pi
    roll_acc = math.atan2(-acc_x, math.sqrt(acc_y ** 2 + acc_z ** 2)) * 180 / math.pi

    # Update pitch and roll using the Kalman filter
    pitch, pitch_bias = kalmanFilterPitch.update(pitch_acc, gyro_x, dt)
    roll, roll_bias = kalmanFilterRoll.update(roll_acc, gyro_y, dt)
    
    return pitch, roll

def update_motor_speeds(pitch_output, roll_output):
    base_speed = 1500  # Base PWM signal for hovering (adjust based on drone)
    
    motor1_speed = base_speed + pitch_output + roll_output
    motor2_speed = base_speed - pitch_output + roll_output
    motor3_speed = base_speed - pitch_output - roll_output
    motor4_speed = base_speed + pitch_output - roll_output
    
    # Normalize to valid PWM range
    motor1_speed = max(1000, min(2000, motor1_speed))
    motor2_speed = max(1000, min(2000, motor2_speed))
    motor3_speed = max(1000, min(2000, motor3_speed))
    motor4_speed = max(1000, min(2000, motor4_speed))

    # Print the motor speeds (replace with actual motor control logic)
    print(f"Motor1: {motor1_speed}, Motor2: {motor2_speed}, Motor3: {motor3_speed}, Motor4: {motor4_speed}")

class DroneControlApp(BoxLayout):
    def __init__(self, **kwargs):
        super(DroneControlApp, self).__init__(**kwargs)
        self.pitch = 0.0
        self.roll = 0.0
        self.previous_time = time.time()
        init_sensors()
        init_gps()  # Initialize the GPS
        Clock.schedule_interval(self.update, 1.0 / 50.0)  # Run at 50 Hz

    def update(self, dt):
        current_time = time.time()
        delta_time = current_time - self.previous_time
        self.previous_time = current_time

        # Read sensor data
        gyro_x, gyro_y, gyro_z = read_gyroscope()
        acc_x, acc_y, acc_z = read_accelerometer()
        latitude, longitude, altitude = get_gps_data()  # Get the latest GPS data

        # Perform sensor fusion to get pitch and roll
        self.pitch, self.roll = sensor_fusion(acc_x, acc_y, acc_z, gyro_x, gyro_y, delta_time, self.pitch, self.roll)

        # Calculate PID outputs for pitch and roll
        pitch_output = pid_pitch(self.pitch)
        roll_output = pid_roll(self.roll)

        # Apply PID outputs to motor control
        update_motor_speeds(pitch_output, roll_output)

        # Optionally print GPS data for debugging
        print(f"GPS Data: Lat {latitude}, Lon {longitude}, Alt {altitude}")

class DroneApp(App):
    def build(self):
        return DroneControlApp()

if __name__ == '__main__':
    DroneApp().run()
