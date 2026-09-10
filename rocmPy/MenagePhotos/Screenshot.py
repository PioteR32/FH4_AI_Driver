import mss
from PIL import Image
import os
import cv2
from datetime import datetime
import numpy as np

def save_screenshot(screenshot:mss.base.ScreenShot, filename,screenshots_dir):
    """Save a screenshot to a file"""
    img_np = np.array(screenshot)
    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    output_path = screenshots_dir + "\\" + filename
    cv2.imwrite(output_path, frame_bgr)


def capture_screenshot(region=None)-> mss.base.ScreenShot:
    """Take a screenshot, optionally of a specific region, and translate it to English."""
    with mss.mss() as sct:
        if region:
            return sct.grab(region)
        else:
            return sct.grab(monitor = 0)
def resize_screenshot(screenshot:mss.base.ScreenShot,size)->cv2.typing.MatLike:
    img = np.array(screenshot)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),size,interpolation=cv2.INTER_NEAREST)

def save_cv2(screenshot:cv2.typing.MatLike,name:str,path:str):
    cv2.imwrite(path+"\\"+name,screenshot)
    
def read_screenshot(screenshots_dir: str, filename: str)-> mss.base.ScreenShot:
    """Read a screenshot from a file"""
    filepath = os.path.join(screenshots_dir, filename)
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    else:
        print(f"File not found: {filepath}")
        return None