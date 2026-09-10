"""
Keyboard Handler for reading keyboard events using the keyboard library
"""

import keyboard
import time
from typing import Callable, Optional

class KeyboardHandler:
    """
    A class to read keyboard events using the keyboard library.
    
    This class provides functionality to:
    - Register callbacks for key presses
    - Read current key states
    - Monitor keyboard events in real-time
    """
    
    def __init__(self):
        """Initialize the keyboard handler."""
        self.is_listening = False
        self.callbacks = {}
        self.pressed_keys = set()
        
    def add_key_callback(self, key: str, callback: Callable[[], None]):
        """
        Add a callback function for a specific key.
        
        Args:
            key (str): The key to monitor (e.g., 'a', 'space', 'ctrl')
            callback (Callable[[], None]): Function to call when key is pressed
        """
        self.callbacks[key] = callback
        
    def remove_key_callback(self, key: str):
        """
        Remove a callback for a specific key.
        
        Args:
            key (str): The key to remove callback from
        """
        if key in self.callbacks:
            del self.callbacks[key]
            
    def start_listening(self):
        """
        Start listening for keyboard events.
        This method runs in the background and monitors all registered callbacks.
        """
        if self.is_listening:
            return
            
        self.is_listening = True
        
        # Register all callbacks
        for key, callback in self.callbacks.items():
            try:
                keyboard.on_press_key(key, lambda e, cb=callback: cb())
            except Exception as e:
                print(f"Error registering callback for {key}: {e}")
                
    def stop_listening(self):
        """
        Stop listening for keyboard events.
        """
        if not self.is_listening:
            return
            
        self.is_listening = False
        # Unregister all callbacks
        for key in list(self.callbacks.keys()):
            try:
                keyboard.unhook_key(key)
            except Exception:
                pass
                
    def is_key_pressed(self, key: str) -> bool:
        """
        Check if a specific key is currently pressed.
        
        Args:
            key (str): The key to check
            
        Returns:
            bool: True if key is pressed, False otherwise
        """
        try:
            return keyboard.is_pressed(key)
        except Exception:
            return False
            
    def get_pressed_keys(self) -> set:
        """
        Get a set of all currently pressed keys.
        
        Returns:
            set: Set of currently pressed keys
        """
        # This is a simplified implementation - for more advanced usage,
        # we might want to track keys in a more sophisticated way
        return self.pressed_keys
        
    def clear_pressed_keys(self):
        """
        Clear the list of pressed keys.
        """
        self.pressed_keys.clear()
        
    def listen_for_key(self, key: str, callback: Optional[Callable[[], None]] = None):
        """
        Listen for a specific key press once.
        
        Args:
            key (str): The key to listen for
            callback (Callable[[], None], optional): Callback to execute when key is pressed
        """
        def _on_key_press(event):
            if event.event_type == keyboard.KEY_DOWN and event.name == key:
                if callback:
                    callback()
                
                
        keyboard.on_press(_on_key_press)
        
    def wait_for_key(self, key: str) -> bool:
        """
        Wait for a specific key to be pressed.
        
        Args:
            key (str): The key to wait for
            
        Returns:
            bool: True when the key is pressed
        """
        try:
            keyboard.wait(key)
            return True
        except Exception as e:
            print(f"Error waiting for key {key}: {e}")
            return False
            
    def simulate_key_press(self, key: str):
        """
        Simulate a key press.
        
        Args:
            key (str): The key to simulate pressing
        """
        try:
            keyboard.press(key)
            keyboard.release(key)
        except Exception as e:
            print(f"Error simulating key press {key}: {e}")
            
    def simulate_key_press_duration(self, key: str, duration: float):
        """
        Simulate a key press for a specific duration.
        
        Args:
            key (str): The key to simulate pressing
            duration (float): Duration in seconds
        """
        try:
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
        except Exception as e:
            print(f"Error simulating key press {key} for {duration}s: {e}")
            
    def close(self):
        """
        Close the keyboard handler and clean up resources.
        """
        self.stop_listening()
        self.pressed_keys.clear()

# Example usage
if __name__ == "__main__":
    # Create a keyboard handler instance
    reader = KeyboardHandler()
    
    # Define a callback function
    def on_space_pressed():
        print("Space key pressed!")
        
    # Add the callback for space key
    reader.add_key_callback('space', on_space_pressed)
    
    print("Keyboard handler  initialized. Press SPACE to test callbacks.")
    print("Press ESC to exit...")
    
    # Start listening
    reader.start_listening()
    
    try:
        # Wait for ESC key to exit
        keyboard.wait('esc')
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        print("Keyboard handler closed.")