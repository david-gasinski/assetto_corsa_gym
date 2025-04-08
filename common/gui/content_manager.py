import pyautogui
import time

class ContentManagerGUI():
    
    """
        To interact with AC and CM properly, the following resolutions must be applied
        
        CM -> Fullscreen
        AC -> 800x600
        
        Also, in CM under Appearance -> Interface, check the option "Do not interfere with windows location and size"
    """
    
    DEFAULT_INTERVAL = 0.5 # interval to wait between operations
    
    # GUI RELATIVE POSITIONS 
    # RATIO OF POS / SIZE
    DRIVE = (0.9758, 0.9757)
    TRACK_CHANGE = (0.1044, 0.1208)
    TRACK_FILTER = (0.3195, 0.0736)
    SELECT_FIRST_TRACK = (0.2957, 0.0972)
    TRACK_CONFIRM = (0.3488, 0.4090)
    CLOSE_AC = (0.5070, 0.6160)
    START_GAME = (0.3660, 0.4194)
    
    def __init__(self):
        self.sW, self.sH = pyautogui.size()
        
    def _get_pos(self, ratio: tuple[float, float]) -> tuple[int, int]:
        """
            Converts a relative position ratio (pos / size) to the actual screen coords.
            Ratios are used to keep the position the same independent of screen size 
        """    
        return (
            int(ratio[0] * self.sW),
            int(ratio[1] * self.sH),
        )

    def _click_pos(self, ratio: tuple[float, float], delay: int = 0, right_click: bool = False):
        """
            Waits delay seconds then clicks the cursor in the position defined by ration
        """
        time.sleep(delay)
        mW, mH = self._get_pos(ratio)
        pyautogui.click(x=mW, y=mH, button='right' if right_click else 'left')
        
    def _print(self, text: str, interval: float = 0.1):
        pyautogui.write(text, interval=interval)
        
    def change_track(self, track: str) -> None:
        """
            Uses GUI operations to switch the track on content manager
            To ensure operations work properly, make CM fullscreen
        """
        time.sleep(self.DEFAULT_INTERVAL)
        
        # select track change
        self._click_pos(self.TRACK_CHANGE)
        time.sleep(self.DEFAULT_INTERVAL)

        # select filter  
        self._click_pos(self.TRACK_FILTER)
        time.sleep(self.DEFAULT_INTERVAL)

        # write track
        self._print(track)
        time.sleep(self.DEFAULT_INTERVAL)

        # submit
        self._click_pos(self.TRACK_CONFIRM)
        time.sleep(self.DEFAULT_INTERVAL)
    
    def launch_ac(self) -> None:
        """
            Starts the game using the "Go" button
        """
        time.sleep(self.DEFAULT_INTERVAL)
        self._click_pos(self.DRIVE)
        
    def start_game(self) -> None:
        """
            Once the game is launched, starts driving
        """
        time.sleep(self.DEFAULT_INTERVAL)
        self._click_pos(self.START_GAME)
        
    def close_ac(self) -> None:
        """
            Closes the game
        """
        time.sleep(self.DEFAULT_INTERVAL)
        pyautogui.press('escape')
        time.sleep(self.DEFAULT_INTERVAL)
        self._click_pos(self.CLOSE_AC)
        time.sleep(self.DEFAULT_INTERVAL)
        pyautogui.press('escape') # closes the menu that appears in CM after AC is closed
        
        
if __name__ == '__main__':
    # select barcelone
    gui = ContentManagerGUI()
    
    gui.change_track("barcelona")
    
    gui.launch_ac()
    time.sleep(10)
    gui.start_game()
    time.sleep(10)
    
    gui.close_ac()
    
            