import time
import smbus

class ArducamPTZ:
    I2C_ADDR = 0x18  # Default I2C address of Arducam PTZ Hat
    REG_PAN = 0x00
    REG_TILT = 0x02
    REG_ZOOM = 0x04
    REG_COMMAND = 0x06
    CMD_RESET = 0x01

    def __init__(self, bus_num=1):
        self.bus = smbus.SMBus(bus_num)
        time.sleep(0.1)

    def init(self):
        """Initialize PTZ by writing to registers (if required)"""
        pass  # Arducam PTZ doesn't require init by default

    def reset(self):
        """Reset PTZ to default position"""
        self.bus.write_byte_data(self.I2C_ADDR, self.REG_COMMAND, self.CMD_RESET)
        time.sleep(1.5)

    def set_pan_angle(self, angle):
        """Set pan angle (0–180)"""
        angle = max(0, min(180, int(angle)))
        self.bus.write_word_data(self.I2C_ADDR, self.REG_PAN, angle)

    def set_tilt_angle(self, angle):
        """Set tilt angle (0–180)"""
        angle = max(0, min(180, int(angle)))
        self.bus.write_word_data(self.I2C_ADDR, self.REG_TILT, angle)

    def set_zoom(self, zoom_level):
        """
        Set zoom level (float 1.0–4.0 typically).
        The actual implementation depends on the PTZ hat’s firmware. 
        Here we multiply by 100 to simulate zoom steps.
        """
        value = int(zoom_level * 100)
        self.bus.write_word_data(self.I2C_ADDR, self.REG_ZOOM, value)
