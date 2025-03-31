import struct
import time

class vJoyLinux:
    def __init__(self, reference=1):
        self.reference = reference
        self.device = None
        self.acquired = False
        self.js_path = "/dev/input/js0"
        self.event_path = "/dev/input/event17"
        
    def open(self):
        """Open the virtual joystick device"""
        try:
            self.device = open(self.event_path, 'wb')
            self.acquired = True
            return True
        except Exception as e:
            print(f"Failed to open vJoy device: {e}")
            return False

    def close(self):
        """Close the virtual joystick device"""
        try:
            if self.device:
                self.device.close()
            self.acquired = False
            return True
        except Exception as e:
            print(f"Failed to close vJoy device: {e}")
            return False

    def generateJoystickPosition(self, 
                               wThrottle=0, wRudder=0, wAileron=0,
                               wAxisX=0, wAxisY=0, wAxisZ=0,
                               wAxisXRot=0, wAxisYRot=0, wAxisZRot=0,
                               wSlider=0, wDial=0, wWheel=0,
                               wAxisVX=0, wAxisVY=0, wAxisVZ=0,
                               wAxisVBRX=0, wAxisVBRY=0, wAxisVBRZ=0,
                               lButtons=0, bHats=0, bHatsEx1=0, bHatsEx2=0, bHatsEx3=0):
        """Generate a joystick position structure compatible with the original vJoy"""
        joyPosFormat = "BlllllllllllllllllllIIII"
        pos = struct.pack(joyPosFormat, self.reference, wThrottle, wRudder,
                         wAileron, wAxisX, wAxisY, wAxisZ, wAxisXRot, wAxisYRot,
                         wAxisZRot, wSlider, wDial, wWheel, wAxisVX, wAxisVY, wAxisVZ,
                         wAxisVBRX, wAxisVBRY, wAxisVBRZ, lButtons, bHats, bHatsEx1, bHatsEx2, bHatsEx3)
        return pos

    def _send_event(self, event_type, code, value):
        """Send an input event to the device"""
        # Use signed integer format for steering events
        if event_type == 0x02:  # EV_REL
            EVENT_FORMAT = 'llHHi'
        else:
            EVENT_FORMAT = 'llHHi'
        event = struct.pack(EVENT_FORMAT, 0, 0, event_type, code, value)
        self.device.write(event)
        self.device.flush()

    def update(self, joystickPosition):
        """Update the joystick state based on the provided position structure"""
        if not self.device or not self.acquired:
            return False

        try:
            # Unpack the joystick position structure
            values = struct.unpack("BlllllllllllllllllllIIII", joystickPosition)
            
            # EV_ABS for absolute axes events
            EV_REL = 0x02  # For steering
            EV_ABS = 0x03
            
            # Axis codes
            ABS_X = 0x00  # Left stick X (Steering)
            REL_X = 0x00   # Steering (now using relative movement)
            ABS_RZ = 0x05 # Right trigger (Throttle)
            ABS_Z = 0x02  # Left trigger (Brake)
            
            # Map steering (wAxisX)
            steer_value = values[4]
            # Map 0-32768 to -65535 to 65535
            scaled_steer = int(((steer_value / 32768) * 2 - 1) * 65535)
            # Ensure the value stays within -65535 to 65535
            scaled_steer = max(-65535, min(65535, scaled_steer))
            self._send_event(EV_ABS, ABS_X, scaled_steer)
            
            # Map throttle (wAxisY) from 0-32768 to 0-255
            throttle_value = int((values[5] / 32768.0) * 255)
            throttle_value = max(0, min(255, throttle_value))
            self._send_event(EV_ABS, ABS_RZ, throttle_value)
            
            # Map brake (wAxisZ) from 0-32768 to 0-255
            brake_value = int((values[6] / 32768.0) * 255)
            brake_value = max(0, min(255, brake_value))
            self._send_event(EV_ABS, ABS_Z, brake_value)
            
            # Send a synchronization event
            self._send_event(0, 0, 0)
            
            return True
        except Exception as e:
            # breakpoint()
            print(f"Failed to update joystick state: {e}")
            return False

def setJoy(valueX, valueY, valueZ, onButtons, scale):
    """
    Set joystick position with correct input ranges:
    valueX: 0.4 (full left) to 1.6 (full right), 1.0 is center
    valueY: 0 (no throttle) to 1 (full throttle)
    valueZ: 0 (no brake) to 1 (full brake)
    scale: typically 16384
    """
    # Map steering from 0.4-1.6 range to 0-32768 range
    # At 0.4 -> 0
    # At 1.0 -> 16384
    # At 1.6 -> 32768
    normalized_x = (valueX - 0.4) / 1.2  # Convert to 0-1 range
    xPos = int(normalized_x * 32768)
    xPos = max(0, min(32768, xPos))
    
    # Map throttle and brake from 0-1 to 0-32768
    yPos = int(valueY * 32768)
    zPos = int(valueZ * 32768)
    
    vjoy = vJoyLinux()
    if vjoy.open():
        try:
            if onButtons != 0:
                joystickPosition = vjoy.generateJoystickPosition(wAxisX=xPos, wAxisY=yPos, wAxisZ=zPos, lButtons=onButtons)
                vjoy.update(joystickPosition)
                time.sleep(0.01)

            joystickPosition = vjoy.generateJoystickPosition(wAxisX=xPos, wAxisY=yPos, wAxisZ=zPos)
            vjoy.update(joystickPosition)
        finally:
            vjoy.close()