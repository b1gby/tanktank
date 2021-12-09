import math
import pygame
import pygame.event as GAME_EVENTS
import pygame.locals as GAME_GLOBALS
import pygame.time as GAME_TIME
import random
import time
import os

mypath = os.path.dirname(os.path.realpath(__file__))

def findDots(n, start, end):
    n -= 1
    dots = [(int(start[0]), int(start[1]))]
    changeRangeX = (end[0] - start[0]) / n
    changeRangeY = (end[1] - start[1]) / n
    for i in range(1, n + 1):
        dots.append((int(start[0] + changeRangeX * i), int(start[1] + changeRangeY * i)))
    return dots


def checkPixelsMove(surface, start, vx, vy, angle, direction):
    if direction == 'backward':
        if angle != 180 and angle != 0:
            angle = -angle
            vx = -vx
            vy = -vy
        else:
            angle = 180 - angle
            vx = -vx
            vy = -vy
    directionStatus = ''
    if angle < 0:
        angle = 360 + angle
    if angle == 0:
        directionStatus = 'right'
    elif 0 < angle < 90:
        directionStatus = 'upRight'
    elif angle == 90:
        directionStatus = 'up'
    elif 90 < angle < 180:
        directionStatus = 'upLeft'
    elif angle == 180:
        directionStatus = 'left'
    elif 180 < angle < 270:
        directionStatus = 'downLeft'
    elif angle == 270:
        directionStatus = 'down'
    elif 270 < angle < 360:
        directionStatus = "downRight"

    if directionStatus == 'right' or directionStatus == 'upRight' or directionStatus == 'downRight':
        for i in range(int(start[0]), int(start[0] + vx + 1)):
            try:
                if surface.get_at((i, int(start[1]))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (i, int(start[1]), 5, 5))
                    return True
            except:
                continue
    if directionStatus == 'left' or directionStatus == 'upLeft' or directionStatus == 'downLeft':
        for i in range(int(start[0]), int(start[0] + vx - 1), -1):
            try:
                if surface.get_at((i, int(start[1]))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (i, int(start[1]), 5, 5))
                    return True
            except:
                continue
    if directionStatus == 'down' or directionStatus == 'downLeft' or directionStatus == 'downRight':
        for i in range(int(start[1]), int(start[1] + vy + 1)):
            try:
                if surface.get_at((int(start[0]), i)) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (int(start[0]), i, 5, 5))
                    return True
            except:
                continue
    if directionStatus == 'up' or directionStatus == 'upLeft' or directionStatus == 'upRight':
        for i in range(int(start[1]), int(start[1] + vy - 1), -1):
            try:
                if surface.get_at((int(start[0]), i)) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (int(start[0]), i, 5, 5))
                    return True
            except:
                continue
    return False


def checkPixelsRotate(surface, start, angle, rotateSpeed, direction):
    angle = 90 + angle
    if direction == 'leftFrontRotate':
        for i in range(1, rotateSpeed + 1):
            x = start[0] + i * math.cos((angle + 2) * math.pi / 180)
            y = start[1] + i * math.sin(-(angle + 2) * math.pi / 180)
            try:
                if surface.get_at((int(x), int(y))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (x, y, 5, 5))
                    return True
            except:
                continue
    elif direction == 'rightFrontRotate':
        for i in range(1, rotateSpeed + 1):
            x = start[0] - i * math.cos((angle + 2) * math.pi / 180)
            y = start[1] - i * math.sin(-(angle + 2) * math.pi / 180)
            try:
                if surface.get_at((int(x), int(y))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (x, y, 5, 5))
                    return True
            except:
                continue
    elif direction == 'rightRearRotate':
        for i in range(1, rotateSpeed + 1):
            x = start[0] - i * math.cos((angle + 2) * math.pi / 180)
            y = start[1] - i * math.sin(-(angle + 2) * math.pi / 180)
            try:
                if surface.get_at((int(x), int(y))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (x, y, 5, 5))
                    return True
            except:
                continue
    elif direction == 'leftRearRotate':
        for i in range(1, rotateSpeed + 1):
            x = start[0] + i * math.cos((angle + 2) * math.pi / 180)
            y = start[1] + i * math.sin(-(angle + 2) * math.pi / 180)
            try:
                if surface.get_at((int(x), int(y))) == (101, 101, 101, 255):
                    pygame.draw.rect(surface, (50, 50, 50), (x, y, 5, 5))
                    return True
            except:
                continue
    return False


class world:
    def loadMaps(self):
        self.maps.append(pygame.image.load(os.path.join(mypath, 'Maps/Map1.png')))

    def chooseMap(self):
        self.Map = self.maps[random.randrange(len(self.maps))]

    def drawMap(self, surface):
        self.chooseMap()
        surface.blit(self.Map, (0, 0))

    def __init__(self):
        self.maps = []
        self.loadMaps()
        self.Map = None


class tank:

    def loadPicture(self, Imagedirectory):
        self.picture = pygame.image.load(os.path.join(mypath, Imagedirectory))

    # rotate left
    def rotate_left(self, surface):
        self.leftRotate = True
        self.rightRotate = False
        collision = False
        while self.collision(surface) == 'leftFrontCollision':
            collision = True
            self.leftRotate = False
            self.backwardDirection = True
            self.move(surface)
            time.sleep(.01)
        else:
            if collision:
                collision = False
                self.backwardDirection = False
                self.leftRotate = True
        while self.collision(surface) == 'rightRearCollision':
            collision = True
            self.leftRotate = False
            self.forwardDirection = True
            self.move(surface)
            time.sleep(.01)
        else:
            if collision:
                collision = False
                self.forwardDirection = False
                self.leftRotate = True
        if not collision:
            self.angle += self.rotateSpeed
        if self.angle >= 360:
            self.angle = self.angle - (360 * int(self.angle / 360))
        self.rouPicture = pygame.transform.rotate(self.picture, self.angle)

    # rotate right
    def rotate_right(self, surface):
        self.leftRotate = False
        self.rightRotate = True
        collision = False
        while self.collision(surface) == 'rightFrontCollision':
            collision = True
            self.rightRotate = False
            self.backwardDirection = True
            self.move(surface)
            time.sleep(.01)
        else:
            if collision:
                collision = False
                self.backwardDirection = False
                self.rightRotate = True
        while self.collision(surface) == 'leftRearCollision':
            collision = True
            self.rightRotate = False
            self.forwardDirection = True
            self.move(surface)
            time.sleep(.01)
        else:
            if collision:
                collision = False
                self.forwardDirection = False
                self.rightRotate = True
        if not collision:
            self.angle -= self.rotateSpeed
        if self.angle <= -360:
            self.angle = self.angle + (360 * int(self.angle / -360))
        self.rouPicture = pygame.transform.rotate(self.picture, self.angle)

    # rotate
    def rotate(self, surface):
        collision = False
        if self.leftRotate:
            while self.collision(surface) == 'leftFrontCollision':
                collision = True
                self.leftRotate = False
                self.backwardDirection = True
                self.move(surface)
                time.sleep(.01)
            else:
                if collision:
                    collision = False
                    self.backwardDirection = False
                    self.leftRotate = True
            while self.collision(surface) == 'rightRearCollision':
                collision = True
                self.leftRotate = False
                self.forwardDirection = True
                self.move(surface)
                time.sleep(.01)
            else:
                if collision:
                    collision = False
                    self.forwardDirection = False
                    self.leftRotate = True
            if not collision:
                self.angle += self.rotateSpeed
            if self.angle >= 360:
                self.angle = self.angle - (360 * int(self.angle / 360))
        if self.rightRotate:
            while self.collision(surface) == 'rightFrontCollision':
                collision = True
                self.rightRotate = False
                self.backwardDirection = True
                self.move(surface)
                time.sleep(.01)
            else:
                if collision:
                    collision = False
                    self.backwardDirection = False
                    self.rightRotate = True
            while self.collision(surface) == 'leftRearCollision':
                collision = True
                self.rightRotate = False
                self.forwardDirection = True
                self.move(surface)
                time.sleep(.01)
            else:
                if collision:
                    collision = False
                    self.forwardDirection = False
                    self.rightRotate = True
            if not collision:
                self.angle -= self.rotateSpeed
            if self.angle <= -360:
                self.angle = self.angle + (360 * int(self.angle / -360))
        # self.move(surface)
        self.rouPicture = pygame.transform.rotate(self.picture, self.angle)



    def drawTank(self, surface):
        if not self.isWracked:
            self.pictureAngle = self.angle
            if self.pictureAngle >= 90:
                self.pictureAngle = self.pictureAngle - (90 * int(self.angle / 90))
            if self.pictureAngle <= -90:
                self.pictureAngle = self.pictureAngle + (90 * int(self.angle / -90))
            if self.pictureAngle < 0:
                self.pictureAngle = -self.pictureAngle
            xyDifference = (self.pictureWidth * math.cos(
                self.pictureAngle * math.pi / 180) + self.pictureWidth * math.sin(
                self.pictureAngle * math.pi / 180))

            surface.blit(self.rouPicture,
                         (
                             self.x - xyDifference / 2,
                             self.y - xyDifference / 2))


    # move forward
    def move_forward(self, surface):
        self.forwardDirection = True
        self.backwardDirection = False
        collision = False
        while self.collision(surface) == 'frontLeftCollision':
            if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                collision = True
                self.forwardDirection = False
                self.rightRotate = True
                self.rotate(surface)
                time.sleep(.01)
            else:
                self.forwardDirection = False
        else:
            if collision:
                collision = False
                self.rightRotate = False
                self.forwardDirection = True
        while self.collision(surface) == 'frontRightCollision':
            if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                collision = True
                self.forwardDirection = False
                self.leftRotate = True
                self.rotate(surface)
                time.sleep(.01)
            else:
                self.forwardDirection = False
        else:
            if collision:
                collision = False
                self.leftRotate = False
                self.forwardDirection = True
        if not self.collision(surface):
            self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
            self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
            self.x += self.vx
            self.y += self.vy

    # move backward
    def move_backward(self, surface):
        self.forwardDirection = False
        self.backwardDirection = True
        collision = False
        while self.collision(surface) == 'rearLeftCollision':
            if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                collision = True
                self.backwardDirection = False
                self.leftRotate = True
                self.rotate(surface)
                time.sleep(.01)
            else:
                self.backwardDirection = False
        else:
            if collision:
                collision = False
                self.leftRotate = False
                self.backwardDirection = True
        while self.collision(surface) == 'rearRightCollision':
            if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                collision = True
                self.backwardDirection = False
                self.rightRotate = True
                self.rotate(surface)
                time.sleep(.01)
            else:
                self.backwardDirection = False
        else:
            if collision:
                collision = False
                self.rightRotate = False
                self.backwardDirection = True
        if not self.collision(surface):
            self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
            self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
            self.x -= self.vx
            self.y -= self.vy

    # move
    def move(self, surface):
        if self.forwardDirection:
            collision = False
            while self.collision(surface) == 'frontLeftCollision':
                if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                    collision = True
                    self.forwardDirection = False
                    self.rightRotate = True
                    self.rotate(surface)
                    time.sleep(.01)
                else:
                    self.forwardDirection = False
            else:
                if collision:
                    collision = False
                    self.rightRotate = False
                    self.forwardDirection = True
            while self.collision(surface) == 'frontRightCollision':
                if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                    collision = True
                    self.forwardDirection = False
                    self.leftRotate = True
                    self.rotate(surface)
                    time.sleep(.01)
                else:
                    self.forwardDirection = False
            else:
                if collision:
                    collision = False
                    self.leftRotate = False
                    self.forwardDirection = True
            if not self.collision(surface):
                self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
                self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
                self.x += self.vx
                self.y += self.vy
        if self.backwardDirection:
            collision = False
            while self.collision(surface) == 'rearLeftCollision':
                if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                    collision = True
                    self.backwardDirection = False
                    self.leftRotate = True
                    self.rotate(surface)
                    time.sleep(.01)
                else:
                    self.backwardDirection = False
            else:
                if collision:
                    collision = False
                    self.leftRotate = False
                    self.backwardDirection = True
            while self.collision(surface) == 'rearRightCollision':
                if self.angle != 0 and self.angle != 90 and self.angle != 180 and self.angle != 270 and self.angle != -90 and self.angle != -180 and self.angle != -270:
                    collision = True
                    self.backwardDirection = False
                    self.rightRotate = True
                    self.rotate(surface)
                    time.sleep(.01)
                else:
                    self.backwardDirection = False
            else:
                if collision:
                    collision = False
                    self.rightRotate = False
                    self.backwardDirection = True
            if not self.collision(surface):
                self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
                self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
                self.x -= self.vx
                self.y -= self.vy

    def fire(self):
        if not self.isWracked and len(self.bullets) < 5:
            bullet(self)

    def collision(self, surface):
        x = 0  # just to debug!
        rightFrontCornerX = self.x + self.width * math.cos((self.angle * math.pi) / 180) / 2 + self.height * math.sin(
            (self.angle * math.pi) / 180) / 2
        rightFrontCornerY = self.y + self.width * math.sin((-self.angle * math.pi) / 180) / 2 + self.height * math.cos(
            (self.angle * math.pi) / 180) / 2
        leftFrontCornerX = rightFrontCornerX - self.height * math.sin((self.angle * math.pi) / 180)
        leftFrontCornerY = rightFrontCornerY - self.height * math.cos((self.angle * math.pi) / 180)
        rightRearCornerX = rightFrontCornerX - self.width * math.cos((self.angle * math.pi) / 180)
        rightRearCornerY = rightFrontCornerY - self.width * math.sin((-self.angle * math.pi) / 180)
        leftRearCornerX = leftFrontCornerX - self.width * math.cos((self.angle * math.pi) / 180)
        leftRearCornerY = leftFrontCornerY - self.width * math.sin((-self.angle * math.pi) / 180)

        frontLeftDots = findDots(25, (leftFrontCornerX, leftFrontCornerY), (
            (leftFrontCornerX + rightFrontCornerX) / 2, (leftFrontCornerY + rightFrontCornerY) / 2))
        frontRightDots = findDots(25, (
            (leftFrontCornerX + rightFrontCornerX) / 2, (leftFrontCornerY + rightFrontCornerY) / 2),
                                  (rightFrontCornerX, rightFrontCornerY))
        rearLeftDots = findDots(25, (leftRearCornerX, leftRearCornerY), (
            (leftRearCornerX + rightRearCornerX) / 2, (leftRearCornerY + rightRearCornerY) / 2))
        rearRightDots = findDots(25, (
            (leftRearCornerX + rightRearCornerX) / 2, (leftRearCornerY + rightRearCornerY) / 2),
                                 (rightRearCornerX, rightRearCornerY))
        leftRearDots = findDots(38, (leftRearCornerX, leftRearCornerY),
                                ((leftFrontCornerX + leftRearCornerX) / 2 + 1,
                                 (leftFrontCornerY + leftRearCornerY) / 2 + 1))
        leftFrontDots = findDots(38,
                                 ((leftRearCornerX + leftFrontCornerX) / 2,
                                  (leftRearCornerY + leftFrontCornerY) / 2),
                                 (leftFrontCornerX, leftFrontCornerY))
        rightRearDots = findDots(38, (rightRearCornerX, rightRearCornerY), (
            (rightFrontCornerX + rightRearCornerX) / 2 + 1, (rightFrontCornerY + rightRearCornerY) / 2 + 1))
        rightFrontDots = findDots(38, (
            (rightRearCornerX + rightFrontCornerX) / 2 + 1, (rightRearCornerY + rightFrontCornerY) / 2 + 1),
                                  (rightFrontCornerX, rightFrontCornerY))

        if self.forwardDirection:
            for dot in frontRightDots:
                if checkPixelsMove(surface, dot, self.vx, self.vy, self.angle, 'forward'):
                    return 'frontRightCollision'
            for dot in frontLeftDots:
                if checkPixelsMove(surface, dot, self.vx, self.vy, self.angle, 'forward'):
                    return 'frontLeftCollision'

        if self.backwardDirection:
            for dot in rearRightDots:
                if checkPixelsMove(surface, dot, self.vx, self.vy, self.angle, 'backward'):
                    return 'rearRightCollision'
            for dot in rearLeftDots:
                if checkPixelsMove(surface, dot, self.vx, self.vy, self.angle, 'backward'):
                    return 'rearLeftCollision'
        if self.leftRotate:
            for dot in leftFrontDots:
                if checkPixelsRotate(surface, dot, self.angle, self.rotateSpeed, 'leftFrontRotate'):
                    return 'leftFrontCollision'
            for dot in rightRearDots:
                if checkPixelsRotate(surface, dot, self.angle, self.rotateSpeed, 'rightRearRotate'):
                    return 'rightRearCollision'
        if self.rightRotate:
            for dot in rightFrontDots:
                if checkPixelsRotate(surface, dot, self.angle, self.rotateSpeed, 'rightFrontRotate'):
                    return 'rightFrontCollision'
            for dot in leftRearDots:
                if checkPixelsRotate(surface, dot, self.angle, self.rotateSpeed, 'leftRearRotate'):
                    return 'leftRearCollision'
        return False

    def restart(self, X, Y):
        self.x = X
        self.y = Y
        self.angle = random.randrange(360)
        self.isWracked = False
        self.wrackTime = 0
        self.bullets = []

    def __init__(self, X, Y, Imagedirectory, surface):
        self.x = X
        self.y = Y
        self.picture = None
        self.rouPicture = None
        self.pictureAngle = 0
        self.loadPicture(Imagedirectory)
        self.height = 30
        self.width = 50
        self.pictureHeight = 30
        self.pictureWidth = 50
        self.angle = random.randrange(360)
        self.rotateSpeed = 10
        self.v = 5
        self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
        self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
        self.forwardDirection = False
        self.backwardDirection = False
        self.leftRotate = False
        self.rightRotate = False
        self.rotate(surface)
        self.bullets = []
        self.isWracked = False
        self.wrackTime = 0
        self.score = 0


class bullet:
    def __init__(self, tank):
        self.tank = tank
        self.x = tank.x + tank.width * math.cos((tank.angle * math.pi) / 180) / 2
        self.y = tank.y + tank.width * math.sin((-tank.angle * math.pi) / 180) / 2
        self.angle = tank.angle
        self.radius = 8
        self.v = 10
        self.vx = self.v * math.cos((self.angle * math.pi) / 180)
        self.vy = self.v * math.sin((self.angle * math.pi) / 180)
        tank.bullets.append(self)
        self.isExpired = False
        self.startTime = GAME_TIME.get_ticks()
        self.expireTime = 30000

    def draw(self, surface):
        if not self.isExpired:
            if (GAME_TIME.get_ticks() - self.startTime) < self.expireTime:
                self.move()
                pygame.draw.circle(surface, (0, 0, 0), (round(self.x), round(self.y)), self.radius, 0)
            else:
                self.isExpired = True
                self.tank.bullets.remove(self)

    def move(self):
        self.vx = self.v * math.cos((-self.angle * math.pi) / 180)
        self.vy = self.v * math.sin((-self.angle * math.pi) / 180)
        self.x += self.vx
        self.y += self.vy

    def collision(self, surface):
        try:
            if surface.get_at((int(self.x + self.radius), int(self.y))) == (101, 101, 101, 255):
                self.angle = 180 - self.angle
            elif surface.get_at((int(self.x - self.radius), int(self.y))) == (101, 101, 101, 255):
                self.angle = 180 - self.angle
            elif surface.get_at((int(self.x), int(self.y + self.radius))) == (101, 101, 101, 255):
                self.angle = - self.angle
            elif surface.get_at((int(self.x), int(self.y - self.radius))) == (101, 101, 101, 255):
                self.angle = - self.angle
                return "WALL COLLISION"
            if surface.get_at((int(self.x + self.radius), int(self.y))) == (255, 158, 116, 255) or surface.get_at(
                    (int(self.x - self.radius), int(self.y))) == (255, 158, 116, 255) or surface.get_at(
                (int(self.x), int(self.y + self.radius))) == (255, 158, 116, 255) or surface.get_at(
                (int(self.x), int(self.y - self.radius))) == (255, 158, 116, 255):
                if GAME_TIME.get_ticks() - self.startTime > 100:
                    return "PURPLE TANK COLLISION"
            if surface.get_at((int(self.x + self.radius), int(self.y))) == (151, 158, 116, 255) or surface.get_at(
                    (int(self.x - self.radius), int(self.y))) == (151, 158, 116, 255) or surface.get_at(
                (int(self.x), int(self.y + self.radius))) == (151, 158, 116, 255) or surface.get_at(
                (int(self.x), int(self.y - self.radius))) == (151, 158, 116, 255):
                if GAME_TIME.get_ticks() - self.startTime > 100:
                    return "GREEN TANK COLLISION"
        except IndexError:
            pass
