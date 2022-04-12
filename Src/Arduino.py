import sys
import glob
import serial
import serial.tools.list_ports
import pyfirmata
import time
from recordWindow import *

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description  # may need tweaking to match new arduinos
        ]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        arduino_ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        arduino_ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    return arduino_ports

def checkConnected(port):
    pin = 3
    board = pyfirmata.Arduino(port)
    for i in range(10):
        board.digital[pin].write(1)
        time.sleep(0.1)  # delays for 5 seconds
        board.digital[pin].write(0)
        time.sleep(0.1)
    board.exit()