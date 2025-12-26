import machine
import utime

# 1. Setup Pins
trig = machine.Pin(5, machine.Pin.OUT)
echo = machine.Pin(12, machine.Pin.IN)

def get_distance():
    # Ensure trigger is low
    trig.value(0)
    utime.sleep_us(5)
    
    # Send a 10us pulse to trigger the sensor
    trig.value(1)
    utime.sleep_us(10)
    trig.value(0)
    
    # Wait for the echo pin to go HIGH (start of pulse)
    while echo.value() == 0:
        pulse_start = utime.ticks_us()
        
    # Wait for the echo pin to go LOW (end of pulse)
    while echo.value() == 1:
        pulse_end = utime.ticks_us()
    
    # Calculate the duration of the pulse
    pulse_duration = utime.ticks_diff(pulse_end, pulse_start)
    
    # Distance = (time * speed of sound) / 2 (for the round trip)
    # Speed of sound is approx 0.0343 cm per microsecond
    distance = (pulse_duration * 0.0343) / 2
    
    return distance

# Main Loop
print("PathPal: Starting Distance Measurement...")

while True:
    try:
        dist = get_distance()
        print("Distance: {:.2f} cm".format(dist))
        
        # Logic for PathPal Alert
        if dist < 30:
            print("ALERT: Obstacle very close!")
            
        utime.sleep(0.5) # Wait half a second between readings
        
    except OSError as e:
        print("Sensor error or out of range")