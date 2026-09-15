"""
Xbox Controller Emulator using vgamepad
A comprehensive Xbox controller emulator built specifically for vgamepad.
"""

import vgamepad as vg

class XboxControllerEmulator:
    """Xbox controller emulator using vgamepad."""
    
    def __init__(self):
        self.gamepad = None
        self.connected = False
        self._connect()
        
    def _connect(self):
        """Initialize the virtual gamepad."""
        try:
            self.gamepad = vg.VX360Gamepad()
            self.connected = True
        except Exception as e:
            print(f"Failed to connect VGamepad: {e}")
            
    def press_button(self, button_name):
        """Press a button using vgamepad."""
        if not self.connected or not self.gamepad:
            return False
            
        try:
            button_map = {
                'a': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'b': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'x': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'y': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'lb': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'rb': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'start': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'back': vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK
            }
            
            if button_name in button_map:
                self.gamepad.press_button(button_map[button_name])
                self.gamepad.update()
                return True
        except Exception as e:
            print(f"Error pressing button {button_name}: {e}")
        return False
        
    def release_button(self, button_name):
        """Release a button using vgamepad."""
        if not self.connected or not self.gamepad:
            return False
            
        try:
            button_map = {
                'a': vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
                'b': vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
                'x': vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
                'y': vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
                'lb': vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
                'rb': vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
                'start': vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
                'back': vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK
            }
            
            if button_name in button_map:
                self.gamepad.release_button(button_map[button_name])
                self.gamepad.update()
                return True
        except Exception as e:
            print(f"Error releasing button {button_name}: {e}")
        return False
        
    def set_left_stick(self, x, y):
        """Set left stick position."""
        if not self.connected or not self.gamepad:
            return False
            
        try:
            self.gamepad.left_joystick(int(x*32766), y)
            self.gamepad.update()
            return True
        except Exception as e:
            print(f"Error setting left stick: {e}")
        return False
        
    def set_right_stick(self, x, y):
        """Set right stick position."""
        if not self.connected or not self.gamepad:
            return False
            
        try:
            self.gamepad.right_joystick(x, y)
            self.gamepad.update()
            return True
        except Exception as e:
            print(f"Error setting right stick: {e}")
        return False
        
    def set_triggers(self, left, right):
        """Set trigger positions."""
        if not self.connected or not self.gamepad:
            return False
            
        try:
            self.gamepad.left_trigger(int(left*255))
            self.gamepad.right_trigger(int(right*255))
            self.gamepad.update()
            return True
        except Exception as e:
            print(f"Error setting triggers: {e}")
        return False