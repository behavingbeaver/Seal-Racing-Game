import cv2
import numpy
import os
import colorsys
import random
import typing
from datetime import datetime
import sys

def overlay_images(bg:numpy.ndarray, img:numpy.ndarray, vertexes:tuple[tuple[int, int], tuple[int, int]]|tuple[int, int]) -> numpy.ndarray:
    # calculate bounding box
    if type(vertexes[0]) == int:
        y1, y2 = vertexes[1] - img.shape[0]//2, vertexes[1] + img.shape[0]//2
        x1, x2 = vertexes[0] - img.shape[1]//2, vertexes[0] + img.shape[1]//2
    else: x1, y1, x2, y2 = vertexes[0][0], vertexes[0][1], vertexes[1][0], vertexes[1][1]

    alpha_s = img[:, :, 3] / 255.0    # get alhpa channel
    alpha_l = 1.0 - alpha_s

    # overlay img on top of bg using alpha channel
    for c in range(0, 3): bg[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])

    return bg

class SEAL:
    def __init__(self, sprite:numpy.ndarray, x:int, y:int, velocity_module:int, velocity_phase:int, name:str, board_size:tuple[int, int]) -> None:
        self._name = name
        self._sprite = sprite
        self.x = x
        self.y = y
        self._module = velocity_module  # store the module so when the direction is changed floating point errors do not propagate while recomputing it with the eulerian distance
        self.velocity = numpy.array([velocity_module*numpy.cos(velocity_phase), velocity_module*numpy.sin(velocity_phase)])
        self.update_bounding_box()
        self.boundires = board_size

    def update_position(self) -> None:
        # double rounding is needed
        self.x = round(numpy.round(self.x + self.velocity[0]))
        self.y = round(numpy.round(self.y + self.velocity[1]))
        # recalculte the bouning box and fit it to the shape
        self.update_bounding_box()
        if self.bounding_box[0][0] < 0: self.x += abs(self.bounding_box[0][0])
        if self.bounding_box[0][1] < 0: self.y += abs(self.bounding_box[0][1])
        if self.bounding_box[1][0] >= self.boundires[0] - 1: self.x -= abs(self.bounding_box[1][0] - self.boundires[0]) + 2
        if self.bounding_box[1][1] >= self.boundires[1] - 1: self.y -= abs(self.bounding_box[1][1] - self.boundires[1]) + 2
        self.update_bounding_box()
    def update_bounding_box(self) -> None:
        self.bounding_box = [[self.x - self._sprite.shape[1] // 2, self.y - self._sprite.shape[0] // 2], [self.x + self._sprite.shape[1] // 2, self.y + self._sprite.shape[0] // 2]]
        # stretch the bounding box to fit the shape (needed for odd sprite's dimensions)
        if self.bounding_box[1][0] - self.bounding_box[0][0] > self._sprite.shape[1]: self.bounding_box[1][0] += 1
        if self.bounding_box[1][0] - self.bounding_box[0][0] < self._sprite.shape[1]: self.bounding_box[0][0] -= 1
        if self.bounding_box[1][1] - self.bounding_box[0][1] > self._sprite.shape[0]: self.bounding_box[1][1] += 1
        if self.bounding_box[1][1] - self.bounding_box[0][1] < self._sprite.shape[0]: self.bounding_box[0][1] -= 1
    def change_direction(self, angle:float) -> None:
        if angle == -1: return
        self.velocity = numpy.array([self._module*numpy.cos(angle), self._module*numpy.sin(angle)])
        self.update_position()  # update the position to get away from the wall and prevent looping

class GAME:
    def __init__(self, board_number: int) -> None:
        self._board_number = board_number
        self._sprites = {"gaming": {}, "winning": {}}
        self._sprites["gaming"] = {_.split(".")[0]: cv2.imread(f"{sys.path[0]}/Sprites/Gaming/" + _, cv2.IMREAD_UNCHANGED) for _ in os.listdir(f"{sys.path[0]}/Sprites/Gaming/")}
        self._sprites["winning"] = {_.split(".")[0]: cv2.imread(f"{sys.path[0]}/Sprites/Winner/" + _, cv2.IMREAD_UNCHANGED) for _ in os.listdir(f"{sys.path[0]}/Sprites/Winner/")}

        self.hit_wall = False   # used to store frame on which seal hits the walls

        self.size = (800, 800)   # x, y : w, h
        self._SPAWING_AREA = [60, 60]
        #self.__obstacle_color = [133, 71, 8]   # brown
        self.__obstacle_color = [100, 100, 100]
        self._START = (-1, -1)
        self._END = (-1, -1)

        self.board : typing.List[int] = list()
        self._seals : typing.List[SEAL] = list()    # define type to help with vscode code suggestion

    def __LoadBoard(self, index: int) -> None:
        # nameing convention: board{index}_{author}.png
        name = [_ for _ in os.listdir(f"{sys.path[0]}/Boards/") if f"board{index}" in _][0] # the programs doesn't know the author so it has to search for the correct index
        self._author = name.split("_")[-1].split(".")[0]

        pic = cv2.imread(f"{sys.path[0]}/Boards/{name}")    # load image
        self.size = pic.shape[:-1][::-1]   # set size of rendered frame
        # create a 2d list rappresenting the status of the board
        for y in range(0, self.size[1]):
            self.board.append([])
            for x in range(0, self.size[0]):
                if all(pic[y][x] - [100, 100, 100] >= 0): self.board[-1].append(1)
                #find start = red
                elif all(pic[y][x] - [0, 0, 100] >= 0):
                    self._START = (x, y)
                    self.board[-1].append(0)
                #find end = blue
                elif all(pic[y][x] - [100, 0, 0] >= 0):
                    self._END = (x, y)
                    self.board[-1].append(0)
                else: self.board[-1].append(0)
        # convert the board to a numpy array for easier and faster area look up
        self.board = numpy.array(self.board)
        if self._START == (-1, -1): raise ValueError("Starting point not detected")
        if self._END == (-1, -1): raise ValueError("Ending point not detected")

    def __CreateBoundries(self) -> numpy.ndarray:
        for y in range(0, len(self.board)):
            for x in range(0, len(self.board[0])):
                if self.board[y][x]: cv2.rectangle(self._obstacles, (x, y), (x + 1, y + 1), (*self.__obstacle_color[::-1], 255), -1)    # draw a 1 pixel rectangle for each wall pixel

    def __DrawBackground(self) -> None:
        # overaly the walls to the background blue-to-white gradient
        self._background = numpy.array([self.size[0] * [365 * numpy.array([*colorsys.hsv_to_rgb(227 / 365, _ / self.size[1], 69 / 100)[::-1], 1/365])] for _ in range(0, self.size[1])], dtype=numpy.uint8)
        self._obstacles = numpy.zeros_like(self._background)

    def initialize(self) -> None:
        print("[DEBUG] loading board")
        self.__LoadBoard(self._board_number)
        print("[DEBUG] drawing background")
        self.__DrawBackground()
        print("[DEBUG] creating boundries")
        self.__CreateBoundries()
        print("[DEBUG] creating seals")
        for _ in self._sprites["gaming"]:
            y = self._sprites["gaming"][_].shape[0] // 2
            x = self._sprites["gaming"][_].shape[1] // 2
            if _ == "END": continue # skip target sprite
            # search for a valid coordinate in which to spawn the seal
            while True:
                rand_x = random.randint(self._START[0] - self._SPAWING_AREA[0], self._START[0] + self._SPAWING_AREA[0])
                rand_y = random.randint(self._START[1] - self._SPAWING_AREA[1], self._START[1] + self._SPAWING_AREA[1])

                if rand_x < x or rand_y < y: continue
                if rand_x + x > self.size[0] or rand_y + y > self.size[1]: continue

                if numpy.sum(self.board[rand_y - y : rand_y + y, rand_x -  x: rand_x + x]) == 0: break
            print("[DEBUG] spawned " + _)
            module = 5  # seal velocity
            angle = numpy.deg2rad(random.randint(0, 365))   # pull a random starting angle
            self._seals.append(SEAL(self._sprites["gaming"][_], rand_x, rand_y, module, angle, _, self.size))
            self.board[self._seals[-1].bounding_box[0][1] : self._seals[-1].bounding_box[1][1], self._seals[-1].bounding_box[0][0] : self._seals[-1].bounding_box[1][0]] = 2    # fill the area covered by the seal with 2s

    def render_next_frame(self) -> numpy.ndarray:
        frame = self._background.copy()  # store backround pic in frame
        # draw spawn area DEBUG ONLY
        # frame = cv2.rectangle(frame, (self._START[0]  - self._SPAWING_AREA[0] , self._START[1]  - self._SPAWING_AREA[1] ), (self._START[0]  + self._SPAWING_AREA[0] , self._START[1]  + self._SPAWING_AREA[1] ), (0, 0, 255), 1)

        ## start scene drawing

        mask = self._obstacles[..., 3] != 0     # create obstacles mask
        frame[mask] = self._obstacles[mask]     # replace all pixels in the mask with the obstacle

        frame = overlay_images(frame, self._sprites["gaming"]["END"], (self._END[0] , self._END[1]))

        ## end scene drawing
        ## start seals update

        # cleaning board
        self.board = numpy.where(self.board > 1, 0, self.board) # clear board from prev. frame seal shadow
        for seal in self._seals:
            seal.update_position()
            # fill the new seal's occupied area with 2 (seal shadow)
            tmp = self.board[seal.bounding_box[0][1] : seal.bounding_box[1][1], seal.bounding_box[0][0] : seal.bounding_box[1][0]]
            tmp = numpy.where(tmp == 0, 2, tmp)
            self.board[seal.bounding_box[0][1] : seal.bounding_box[1][1], seal.bounding_box[0][0] : seal.bounding_box[1][0]] = tmp

        # bouncing iteration
        for seal in self._seals:
            # draw seal bounding box DEBUG ONLY
            ## frame = cv2.rectangle(frame, (seal.bounding_box[0][0] , seal.bounding_box[0][1]),
            ##                              (seal.bounding_box[1][0] - 1, seal.bounding_box[1][1] - 1), (0, 255, 0), 1)

            x1 = seal.bounding_box[0][0]
            x2 = seal.bounding_box[1][0]
            y1 = seal.bounding_box[0][1]
            y2 = seal.bounding_box[1][1]

            # check if the end point is inside the bounding box
            if x1 < self._END[0] < x2 and y1 < self._END[1] < y2:
                self.winner = seal._name
                return 1

            angle = -1
            normalization_factor = max(self.board[y1: y2, x1: x2].shape) + .1
            # calculate the ammount of area (normalised) that is in contact with a wall or another seal
            north = numpy.sum(self.board[y1 - 1, x1: x2]) / normalization_factor # +1 to avoid divisions by 0
            sud   = numpy.sum(self.board[y2 + 1, x1: x2]) / normalization_factor
            est   = numpy.sum(self.board[y1: y2, x2 + 1]) / normalization_factor
            west  = numpy.sum(self.board[y1: y2, x1 - 1]) / normalization_factor

            # the bouncing is only in the 4 cardinal direction we ignore the in betweens
            # the seal will bounce of the direction that sees the most contact with a foreign object
            max_dir = max(north, sud, est, west)
            max_dir = max_dir if max_dir > 0 else -1

            if north  == max_dir: angle = numpy.deg2rad(random.randint(30, 150)) # the angle is technically switched with sud's one but it was doing the opposite so idk
            elif sud  == max_dir: angle = numpy.deg2rad(random.randint(150, 335))
            elif est  == max_dir: angle = numpy.deg2rad(random.randint(120, 245))
            elif west == max_dir: angle = numpy.deg2rad(180 - random.randint(120, 245))

            if angle != -1: self.hit_wall = True

            seal.change_direction(angle)

        ## end seals update
        ## start seals drawing

        for seal in self._seals: frame = overlay_images(frame, seal._sprite, seal.bounding_box)

        ## end seals drawing

        return frame

class RECORDER:
    def __init__(self, board_number:int, display = True) -> None:
        self.board_number = board_number
        self.display = display
        self.game = GAME(self.board_number)
        self.game.initialize()
        
        self.game_frames = []
        self.hits = []   # store frame on wich a seal hits a wall
        self.fps = 30
        self.intro_seconds = 5
        self.n = 0

    def __display_frame(self, frame, frame_rate) -> None:
        if not self.display: return
        cv2.imshow("game", frame)
        cv2.waitKey(frame_rate)

    def StartingScreen(self) -> None:
        # starting screen
        starting_frame = self.game.render_next_frame()
        # generete first frame to use as base
        number_of_frames = self.fps*self.intro_seconds
        for self.n in range(0, number_of_frames):
            # x position of the bounching screen (20 to offset for the negative distance in the rectagle drawing)
            x = round(20 + self.game.size[0] * self.n / number_of_frames)
            # if the x coordinate is grater than the window (340 adjust for the width of the text) than subract from the end (-1= last position, etc.)
            # (game.size[0] - 680) is a correcting factor to the bounce idk where it stems from but it's linear
            x = self.game.size[0] - x + (self.game.size[0] - 680) if x + 340 > self.game.size[0] else x

            # x position of the bounching screen (35 to offset for the negative distance in the rectagle drawing)
            y = round(35 + self.game.size[1] * self.n / number_of_frames)
            # if the y coordinate is grater than the window (35 adjust for the height of the text) than subract from the end (-1= last position, etc.)
            # game.size[1] is multiplied by 2 cause negative values are seen as negative coordinates and not negative indexes (so 1 to "normalise" the counter and the second to get the actual position)
            y = 2 * self.game.size[1] - y - 35 if y + 20 > self.game.size[1] else y

            # draw red rectangle and than the text on top
            frame = cv2.rectangle(starting_frame.copy(), (x - 20, y + 20), (x + 340, y - 35), (0,0,255), -1)
            # '%.2f' %  keeps the last 0
            frame = cv2.putText(frame, f"Place your bets in: {'%.2f' % round(self.intro_seconds - self.n/30, 2)}s", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)

            self.game_frames.append(frame.copy()[:,:,:3])

            self.__display_frame(frame, 1000//30)

    def GameScreen(self, ending=True) -> None:
        while True:
            frame = self.game.render_next_frame()
            if type(frame) == int:
                if ending: self.EndScreen()
                break
            else:
                self.game_frames.append(frame.copy()[:,:,:3])
                if self.game.hit_wall:
                    self.hits.append(self.n)
                    self.game.hit_wall = False
                self.n += 1
                # live view 3x speed
                self.__display_frame(frame, 1)  #1000//12

    def EndScreen(self) -> None:
        # pre compute static parts of the ending screen to speed up the calculation of the frame
        frame = self.game._background
        frame = cv2.putText(frame, self.game.winner, (400,250), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 4)
        frame = overlay_images(frame, self.game._sprites["winning"][self.game.winner], (400, 350))
        # credits
        frame = cv2.putText(frame, "board by @" + self.game._author, (0, self.game.size[1]-70), cv2.FONT_ITALIC, 0.5, (0,0,0), 1)
        frame = cv2.putText(frame, "game by @behavingbeaver", (0, self.game.size[1]-45), cv2.FONT_ITALIC, 0.5, (0,0,0), 1)
        frame = cv2.putText(frame, "sprites work by @Leptonychotes", (0, self.game.size[1]-20), cv2.FONT_ITALIC, 0.5, (0,0,0), 1)

        # make the text THE WINNER IS rainbow color it: times the rainbow is runned throw, c: color of the rainbow
        for it in range(0, 5):
            for c in range (0, 30):
                frame = cv2.putText(frame, "THE WINNER IS", (300,200), cv2.FONT_HERSHEY_COMPLEX, 1, numpy.array(colorsys.hsv_to_rgb(c / 30, 1, 1)[::-1]) * 255, 4)
                self.game_frames.append(frame.copy()[:,:,:3])
                self.__display_frame(frame, 1000//30)
                self.n += 1
    
    def SaveRecording(self) -> None:
        # create silent audio track as long as the intro + race + ending screeing (n is the counter)
        os.system(f"ffmpeg -hide_banner -loglevel error -f lavfi -i anullsrc=r=11025:cl=mono -t {self.n // self.fps} -acodec aac {sys.path[0]}/base.wav")

        # start of the ffmpeg filter
        # command example: ffmpeg -i base.wav -i sound_effect.wav -filter_complex "[1]adelay=100[b];[1]adelay=100[c];[0][b][c]amix=3" test.wav
        print("[DEBUG] creating audio track")
        command = f'ffmpeg -hide_banner -loglevel error -i {sys.path[0]}/base.wav -i {sys.path[0]}/sound_effect.wav -filter_complex "'

        names = ["0"]   # keep track of name generated

        print("[DEBUG] creating ffmpeg command")
        for h in self.hits:
            i = self.hits.index(h) + 1
            names.append("".join([chr((ord('a')+i % 26 if not _ else ord('`')+_)) for _ in range(i // 26, -1, -1)]))    # generate letters from numbers: 1->a, 26->z, 27->aa, ecc
            command += f"[1]adelay={round((h/self.fps)*1000)}[{names[-1]}];"

        command += "[" + "][".join(names) + f']amix={len(names)}" {sys.path[0]}/audio_low.wav'

        print("[DEBUG] running ffmpeg command")
        os.system(command)
        os.system(f'ffmpeg -hide_banner -loglevel error -i {sys.path[0]}/audio_low.wav -filter:a "volume=15dB" {sys.path[0]}/audio.wav')    # amplify audio because ffmpeg reduce the volume to a fix hight
        os.remove(f"{sys.path[0]}/base.wav")        # remove background sound
        os.remove(f"{sys.path[0]}/audio_low.wav")   # remove low volume audio 

        print("[DEBUG] creating video")
        # save frames to video called game.mp44 with mp4v codec
        out = cv2.VideoWriter(f"{sys.path[0]}/game.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.game.size)
        for frame in self.game_frames: out.write(frame)
        out.release()
        # use ffmpeg to convert video from mp4v codec (not supported by twitter) to h.264 (supported) and delete the incorrect codec video
        print("[DEBUG] converting video")
        os.system(f"ffmpeg -hide_banner -loglevel error -i {sys.path[0]}/game.mp4 -an -vcodec libx264 -crf 23 {sys.path[0]}/game_tmp.mp4")
        os.remove(f"{sys.path[0]}/game.mp4")
        # add audio to video and remove both of them
        print("[DEBUG] merging video and audio")
        os.system(f"ffmpeg -hide_banner -loglevel error -i {sys.path[0]}/game_tmp.mp4 -i {sys.path[0]}/audio.wav -c:v copy -c:a aac {sys.path[0]}/game_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}_board{board_number}.mp4")
        os.remove(f"{sys.path[0]}/game_tmp.mp4")
        os.remove(f"{sys.path[0]}/audio.wav")

    def PLAY(self) -> None:
        self.StartingScreen()
        self.GameScreen()
        self.SaveRecording()

if __name__ == "__main__":
    board_number = sys.argv[1] if len(sys.argv) > 1 else input("Board number: ")
    mode = "n" if len(sys.argv) < 3 else sys.argv[2]
    rec = RECORDER(board_number)
    # modes:
    #   n: normal
    #   b: bulk (used to stress stest the physics engine)
    #   w: winner (search for a designated winner)
    match mode:
        case "n": rec.PLAY()
        case "b": 
            for _ in range(0, 100):
                try:
                    rec.display = False
                    rec.GameScreen(ending=False)
                except KeyboardInterrupt: pass  # stop current simulation
                except ValueError:  # the engine failed and a sprite was drawm outside the boundires
                    cv2.imshow("a", rec.game_frames[-1])    # show last frame to visualy assess problem
                    cv2.waitKey()
        case "w":
            if len(sys.argv) < 4: raise TypeError("w mode require the name of the winner to be searched")
            winner = sys.argv[3]
            while True:
                print("[DEBUG] RUNNING")
                rec.display = False
                rec.StartingScreen()
                rec.GameScreen()
                print("[DEBUG] game winner", rec.game.winner)
                if rec.game.winner == winner:   # repete until designated winner is found
                    rec.SaveRecording()
                    break
    