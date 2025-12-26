import machine
import utime

# 1. Setup Pins
trig = machine.Pin(5, machine.Pin.OUT)
echo = machine.Pin(12, machine.Pin.IN)

def get_distance():
    # Ensure trigger is low
    trig.value(0)
    utime.sleep_us(5)
    
    # Send pulse
    trig.value(1)
    utime.sleep_us(10)
    trig.value(0)
    
    # Measure the pulse duration in microseconds
    # 30000us (30ms) is the timeout; if no echo, it returns -1 or -2
    duration = machine.time_pulse_us(echo, 1, 30000)
    
    if duration < 0:
        return None  # This handles the "hanging" issue
        
    distance = (duration * 0.0343) / 2
    return distance

# Main Loop
print("PathPal: Starting Distance Measurement...")

while True:
    try:
        dist = get_distance()
        if dist is None:
            continue
        else:
            print("Distance: {:.2f} cm".format(dist))
        
            # Logic for PathPal Alert
            if dist < 30:
                print("ALERT: Obstacle very close!")
            
        utime.sleep(0.5) # Wait half a second between readings
        
    except OSError as e:
        print("Sensor error or out of range")