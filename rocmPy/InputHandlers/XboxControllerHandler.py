
import time
import XInput

class XboxControllerReader:
    def __init__(self):
        pass
    def read_controller_state(self):
        try:
        # Pobieramy pełny stan pierwszego kontrolera (index 0)
            state = XInput.get_state(0)

        # 1. ODCZYT TRIGGERÓW (LT / RT) -> Zakres: 0.0 do 1.0
            lt = XInput.get_trigger_values(state)[0]  # Left Trigger (LT)
            rt = XInput.get_trigger_values(state)[1]  # Right Trigger (RT)

        # 2. ODCZYT GAŁEK ANALOGOWYCH -> Zakres: -1.0 do 1.0 dla osi X i Y
        # (x_pos: -1=lewo, 1=prawo | y_pos: -1=dół, 1=góra)
            lx = XInput.get_thumb_values(state)[0][0]  # Krotka (x, y)
            return lt, rt, lx
        

        except KeyboardInterrupt:
            print("\nZakończono.")
