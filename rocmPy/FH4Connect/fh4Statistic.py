import math
import socket
import struct
import time

import threading
class FH4TelemetryListener:
    def __init__(self, ip="127.0.0.1", port=6789):
        self.UDP_IP = ip
        self.UDP_PORT = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.data = None
        self.is_running = True
        self.lock = threading.Lock()
        print(f"Listening for FH4 telemetry on {self.UDP_IP}:{self.UDP_PORT}...")
        threading.Thread(target=self.listen_for_telemetry).start()

    def listen_for_telemetry(self):
        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if len(data) == 324:
                    with self.lock:
                        self.data = data
                        
            except KeyboardInterrupt:
                print("\n\nZatrzymano nasłuchiwanie.")
                break

    def get_speed_from_telemetry(self)->float:
        try: 
            if self.data is None:
                return None
            with self.lock:
                vx, vy, vz = struct.unpack("<fff", self.data[32:44])
            return math.sqrt(vx**2 + vy**2 + vz**2) * 3.6

        except KeyboardInterrupt:
            print("\n\nZatrzymano nasłuchiwanie.")

    def get_race_position_from_telemetry(self):
        try: 
            if self.data is None:
                return None
            with self.lock:
                race_position = struct.unpack_from('B', self.data, 314)[0]
            return race_position
        except KeyboardInterrupt:
            print("\n\nZatrzymano nasłuchiwanie.")
    #160-164 jak wyjeżdża na trawę to się zwiększa normalnie 0.0
    def get_tire_split(self):
        try:
            if self.data is None:
                return None
            with self.lock:
                slip_ratio = struct.unpack_from("<ffff", self.data, 212)    # Uślizg wzdłużny
                slip_angle = struct.unpack_from("<ffff", self.data, 228)    # Uślizg poprzeczny
                combined_slip = struct.unpack_from("<ffff", self.data, 244) # Uślizg złożony
                norm_slip = struct.unpack_from("<ffff", self.data, 144)

                # Offset 160: Tire Combined Slip (FL, FR, RL, RR)
                comb_slip = struct.unpack_from("<ffff", self.data, 160)

                print(f"Normalized Slip (144): {norm_slip}")
                print(f"Combined Slip   (160): {comb_slip}")
                # Prędkość obrotowa kół z offsetu 80
                wheel_speed = struct.unpack_from("<ffff", self.data, 80)

                print(f"Wheel Speed (80-95):  {[round(x, 1) for x in wheel_speed]}")
                print(f"Slip Ratio (212-227): {[round(x, 2) for x in slip_ratio]}")
                print(f"Combined   (244-259): {[round(x, 2) for x in combined_slip]}")
                print("=" * 60)
                
        except KeyboardInterrupt:
            print("\n\nZatrzymano nasłuchiwanie.")
    def __del__(self):
        self.is_running=False
        self.sock.close()
    #test    
if __name__ == "__main__":
    listener = FH4TelemetryListener()
    while True:
        speed = listener.get_speed_from_telemetry()
        if speed is not None:
            print(f"Current Speed: {speed:.2f} km/h")
        race_position = listener.get_race_position_from_telemetry()
        if race_position is not None:
            print(f"Current Race Position: {race_position}")
        listener.get_tire_split()
        time.sleep(0.1)  # Dodajemy krótką przerwę, aby nie przeciążać CPU