import ML
import FSM
import serial

UPPERBOUND = 60.05
LOWERBOUND = 59.95

ser = serial.Serial(
    port=serial.tools.list_ports.comports()[len() - 1],
    baudrate=115200
)

if __name__ == "__main__":
    freq = ser.read()
    while  freq < UPPERBOUND or freq > LOWERBOUND:
        ML.train()
        freq = ser.read()
    
