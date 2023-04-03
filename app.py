import os
import random
import time

import cv2
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp.drawing = mp.solutions.drawing_utils

from utils import *


class Pipe:
    def __init__(self, x, path='assests/pipe/pipe.png'):
        self.x = x
        self.height = np.random.randint(150, 450)
        self.top_y = 0
        self.bottom_y = self.height + PIPE_GAP
        self.path = path


    def draw(self, frame):
        # pipe_top, pipe_top_mask = read_bgra(self.path, width=PIPE_WIDTH, height=self.height)
        # rollback(frame, pipe_top_mask, pipe_top, self.x, self.top_y, self.x + PIPE_WIDTH, self.top_y + self.height)


        cv2.rectangle(frame, (self.x, self.top_y), (self.x + PIPE_WIDTH, self.top_y + self.height), random.choice(COLORS), thickness=2)
        cv2.rectangle(frame, (self.x, self.bottom_y), (self.x + PIPE_WIDTH, self.bottom_y + WINDOW_HEIGHT), random.choice(COLORS), thickness=2)

    def update(self):
        self.x -= PIPE_SPEED

    def check_collision(self, bird):
        if bird.x + BIRD_WIDTH > self.x and bird.x < self.x + PIPE_WIDTH:
            if bird.y < self.height or bird.y + BIRD_HEIGHT > self.bottom_y:
                return True
        return False


class Ranking:
    def __init__(self, ranking_store='ranking.txt'):
        self.ranking_store = ranking_store

    def log(self, name, score):

        with open(self.ranking_store, 'a') as f:
            f.write(f"{name}, {score}\n")

    def show(self, frame, limit=5):
        cv2.putText(frame, 'RANKING SYSTEM', org=(1000, 40), fontScale=1, fontFace=2, color=(255, 255, 255),
                    thickness=2)
        with open(self.ranking_store, 'r') as f:
            data = [line.rstrip('\n') for line in f]

        if len(data) > 1:
            data.sort(key=lambda x: str(x.split(',')[1]), reverse=True)

        if len(data) < limit:
            limit = len(data)

        for id, user in enumerate(data[:limit]):
            name, score = user.split(',')
            cv2.putText(frame, f'{name.ljust(8)} {score}', org=(1000, 80 + id * 40), fontScale=0.8, fontFace=2,
                        color=(10, 255, 255),
                        thickness=1)

    def reset(self):
        os.remove(self.ranking_store)


class Bird:
    def __init__(self, center):
        self.x = center[0]
        self.y = center[1]
        self.velocity = 0

    def draw(self, frame, id_bird=0):
        path = f'assests/bird/bird_{id_bird}.png'
        bird_img, mask = read_bgra(path, BIRD_WIDTH, BIRD_HEIGHT)

        h, w = bird_img.shape[:2]
        rollback(frame, mask, bird_img, x1=self.x, x2=self.x + w, y1=self.y, y2=self.y + h)

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def jump(self):
        self.velocity = -JUMP_VELOCITY


class FaceTracking:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

    def get_face_center(self, results):
        faces = []
        for detection in results.detections:
            faces.append((detection.location_data.relative_keypoints[2].x,
                          detection.location_data.relative_keypoints[2].y,
                          detection.score[0]))

        faces = faces[np.argmax(np.array(faces)[:, 2])][:2]
        return faces


    def tracking(self, image):
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)
        center = self.get_face_center(results)
        center_x, center_y = int(center[0] * w), int(center[1] * h)
        return center_x, center_y


pipes = [Pipe(WINDOW_WIDTH + i * (PIPE_WIDTH + 200)) for i in range(PIPE_NUMBER)]


def main():
    score = 0
    game_over = False
    id_bird = 0
    pause = -1

    face_tracking = FaceTracking()
    ranking = Ranking()
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        center = face_tracking.tracking(frame)
        bird = Bird(center)

        bird.draw(frame, id_bird=id_bird)
        id_bird = 0 if id_bird == 3 else id_bird + 1

        for pipe in pipes:
            pipe.draw(frame)

            if pipe.x < -PIPE_WIDTH:
                pipes.remove(pipe)
                pipes.append(Pipe(WINDOW_WIDTH))

            if pause == -1:
                if not game_over:
                    score += 1
                    pipe.update()

                if pipe.check_collision(bird):
                    game_over = True

        cv2.putText(frame, text=f"Score: {score}", org=(30, 30), fontFace=1, fontScale=2, color=(128, 128, 255),
                    thickness=2)
        ranking.show(frame)

        if game_over:
            cv2.putText(frame, text="GAME OVER", org=(500, 300), fontScale=2, fontFace=2,
                        color=(0, 0, 255), thickness=3)
            print(RED + "GAME OVER!" + NOCOLOR)
            player = input(YELLOW + "Enter your name bro: " + NOCOLOR)
            ranking.log(player, score)
            print(LIGHTBLUE + f"Restart app to play again, {player}" + NOCOLOR)
            time.sleep(1)
            break


        cv2.imshow('FLIPPY BIRD', frame)

        key = cv2.waitKey(1)

        if key == ord('p'):
            pause = 0 - pause

        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
