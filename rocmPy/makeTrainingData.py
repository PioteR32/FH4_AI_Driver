from typing import List

from FH4Connect.fh4Statistic import FH4TelemetryListener
from InputHandlers.XboxControllerHandler import XboxControllerReader
from InputHandlers.KeyboardHandler import KeyboardHandler
from multiprocessing import  Queue
from multiprocessing import  Event
from pathlib import Path

import MenagePhotos.Screenshot as Screenshot
import multiprocessing
import bettercam
import shutil
import time
import csv
import cv2
import os


crop = (int)(150), (int)(190), (int)(2050), (int)(680) #left, top, right, bottom
region_crop = 0,0,2560,1440
resize = (390,98)
fps = 60
isEnd = False


class SendingClass:
    def __init__(self,img,telemetry):
        self.img = img
        self.telemetry = telemetry
    
crop_2560x1440p = (1280,720)

def save_data(queue:Queue,
               screenshots_dir:str,
               start_index:int=0,
               index_step:int=1,
               stop_event = None
               ):
    i = start_index
    global isEnd
    try:
        string =screenshots_dir + "\\telemetry.csv"
        with open(string,mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["screnshot_name","LT" ,"RT","Steering","Speed_kmh", "Race_Position"])
            print(f"Rozpoczęto zapis danych do pliku telemetry.csv...")
            while True:
                stopwatch = time.perf_counter()
               
                if(not queue.empty()):
                    screenshot = queue.get()
                else:
                    screenshot = None
                if screenshot is not None:
                    i+=index_step
                    screenshot.telemetry[0] = f"screenshot_{i}.png";
                    resized =cv2.resize(cv2.cvtColor(screenshot.img, cv2.COLOR_BGRA2BGR),crop_2560x1440p,interpolation=cv2.INTER_NEAREST)
                    Screenshot.save_cv2(resized,screenshot.telemetry[0],screenshots_dir)
                    writer.writerow(screenshot.telemetry)
                    if(i %100):
                        csv_file.flush()
                elif stop_event is not None and stop_event.is_set():
                   
                    return
                stopwatch_end = time.perf_counter()
                
    except Exception as e:
        print("Exception" + str(e.args))

def takingData(controllerHandler:XboxControllerReader
               ,keyboardHandler:KeyboardHandler
               ,queues:List[Queue]
               ,fps:int
               , crop:tuple
               ,stop_event = None):
    
    camera = bettercam.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=90, video_mode=True)
    telemetry =  FH4TelemetryListener()
    num_queues = len(queues)
    actually_queue = 0 
    print("Click p")
    while not keyboardHandler.is_key_pressed("p"):
        pass
    while keyboardHandler.is_key_pressed("q") == False:
        telemetry_list = [0,0,0,0,0,0]
        start_timestamp_ms = time.perf_counter() 
        telemetry_list[4] = telemetry.get_speed_from_telemetry()
        telemetry_list[5] = telemetry.get_race_position_from_telemetry()
        telemetry_list[1],telemetry_list[2],telemetry_list[3] = controllerHandler.read_controller_state()
        camera_img = camera.get_latest_frame()
        queues[actually_queue].put(SendingClass(camera_img, telemetry_list))
        actually_queue = (actually_queue + 1) % num_queues
        end_timestamp_ms = time.perf_counter()
        # print(f"Screenshot taken time: {(end_timestamp_ms - start_timestamp_ms)} seconds. Time taken: {end_timestamp_ms - start_timestamp_ms} seconds.")
        if(1/fps - (end_timestamp_ms - start_timestamp_ms) > 0):
            time.sleep(1/fps - (end_timestamp_ms - start_timestamp_ms))
    if stop_event is not None:
        stop_event.set()
        return

#390x98
if __name__ == "__main__":

    controllerHandler = XboxControllerReader()
    handler = KeyboardHandler()
    queue = Queue()
    queues = [Queue() for _ in range(5)]  # Example: 5 queues
    screenshots_dir = "c:\\screenshots\\ASTON_MARTIN_FHEDITION_EVAL"
    
    print("Taking screenshots...")
    screenshot_dirs = []
    saving_processes = list()

    stop_event = Event()
    taking_process = multiprocessing.Process(target=takingData, args=(controllerHandler,handler,queues,fps, crop, stop_event))
    for q_idx in range(len(queues)):
        
        i = 0
        while os.path.exists(screenshots_dir + f"_{i}"):
            i+=1
        screenshots_dir = f"{screenshots_dir}_{i}"
        os.makedirs(screenshots_dir)
        screenshot_dirs.append(screenshots_dir)
        saving_process = multiprocessing.Process(
            target=save_data,
            args=(queues[q_idx],
             screenshots_dir,
             q_idx, 
             len(queues),
             stop_event))
        saving_process.start()
        saving_processes.append(saving_process)
   
    taking_process.start()
   

    print("End of taking screenshots...")
    while not len(saving_processes) == 0:
        i = 0
        while i < len(saving_processes):
            
            if not saving_processes[i].is_alive():
                print("remove")
                saving_processes.remove(saving_processes[i])
            else:
                i+=1

    time.sleep(0.5)
    print("End of saving screenshots...")
    with open(screenshot_dirs[0] + "\\telemetry.csv",'a') as main_file:
        for dir in range(1,len(screenshot_dirs)):
            for file_name in os.listdir(screenshot_dirs[dir]):
                if file_name[0] == 's':
                    shutil.move(os.path.join(screenshot_dirs[dir], file_name), screenshot_dirs[0])
                if file_name[0] == 't':
                    with open(screenshot_dirs[dir] + "\\" + file_name,"r") as f:
                        data = f.readlines()[1:]
                        main_file.writelines(data)
            shutil.rmtree(screenshot_dirs[dir])

    #To one Folder function
