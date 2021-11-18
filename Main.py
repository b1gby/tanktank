import random
import sys
import time

import Objects
import pygame
import pygame.event as GAME_EVENTS
import pygame.locals as GAME_GLOBALS
import pygame.time as GAME_TIME

positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
greenTankPosition = positions.pop(random.randrange(len(positions)))
purpleTankPosition = positions.pop(random.randrange(len(positions)))
positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]

pygame.init()
world = Objects.world()
surface = pygame.display.set_mode((1366, 768), pygame.FULLSCREEN)
greenTankPictureDirectory = 'Tanks/greenTank3.png'
greenTank = Objects.tank(greenTankPosition[0], greenTankPosition[1], greenTankPictureDirectory, surface)
purpleTankPictureDirectory = 'Tanks/purpleTank.png'
purpleTank = Objects.tank(purpleTankPosition[0], purpleTankPosition[1], purpleTankPictureDirectory, surface)
bg = pygame.image.load('Backgrounds/tankbg1.png')
bg1 = pygame.image.load('Backgrounds/tankbg.png')
# joystick0=pygame.joystick.Joystick(0)
#joysticks = []
#for i in range(0, pygame.joystick.get_count()):
    #joysticks.append(pygame.joystick.Joystick(i))
    #joysticks[-1].init()


def quitGame():
    pygame.quit()
    sys.exit()


def startedgame():
    while True:
        if greenTank.isWracked and (not purpleTank.isWracked) and GAME_TIME.get_ticks() - greenTank.wrackTime > 5000:
            purpleTank.score += 1
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTankPosition = positions.pop(random.randrange(len(positions)))
            purpleTankPosition = positions.pop(random.randrange(len(positions)))
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTank.restart(greenTankPosition[0], greenTankPosition[1])
            purpleTank.restart(purpleTankPosition[0], purpleTankPosition[1])
        if purpleTank.isWracked and (not greenTank.isWracked) and GAME_TIME.get_ticks() - purpleTank.wrackTime > 5000:
            greenTank.score += 1
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTankPosition = positions.pop(random.randrange(len(positions)))
            purpleTankPosition = positions.pop(random.randrange(len(positions)))
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTank.restart(greenTankPosition[0], greenTankPosition[1])
            purpleTank.restart(purpleTankPosition[0], purpleTankPosition[1])
        if purpleTank.isWracked and greenTank.isWracked:
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTankPosition = positions.pop(random.randrange(len(positions)))
            purpleTankPosition = positions.pop(random.randrange(len(positions)))
            positions = [(93, 50), (1273, 670), (93, 450), (1273, 270), (493, 450), (973, 160), (343, 200), (973, 500)]
            greenTank.restart(greenTankPosition[0], greenTankPosition[1])
            purpleTank.restart(purpleTankPosition[0], purpleTankPosition[1])
        for event in GAME_EVENTS.get():
            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_UP:
                    greenTank.forwardDirection = True
                    greenTank.backwardDirection = False
                if event.key == pygame.K_DOWN:
                    greenTank.forwardDirection = False
                    greenTank.backwardDirection = True
                if event.key == pygame.K_LEFT:
                    greenTank.leftRotate = True
                    greenTank.rightRotate = False
                if event.key == pygame.K_RIGHT:
                    greenTank.leftRotate = False
                    greenTank.rightRotate = True
                if event.key == pygame.K_RCTRL:
                    greenTank.fire()
                # purpleTank

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    purpleTank.forwardDirection = True
                    purpleTank.backwardDirection = False
                if event.key == pygame.K_d:
                    purpleTank.forwardDirection = False
                    purpleTank.backwardDirection = True
                if event.key == pygame.K_s:
                    purpleTank.leftRotate = True
                    purpleTank.rightRotate = False
                if event.key == pygame.K_f:
                    purpleTank.leftRotate = False
                    purpleTank.rightRotate = True
                if event.key == pygame.K_a:
                    purpleTank.fire()
                if event.key == pygame.K_ESCAPE:
                    pausemenu()

            if event.type == pygame.KEYUP:
                # greenTank
                if event.key == pygame.K_UP:
                    greenTank.forwardDirection = False
                if event.key == pygame.K_DOWN:
                    greenTank.backwardDirection = False
                if event.key == pygame.K_LEFT:
                    greenTank.leftRotate = False
                if event.key == pygame.K_RIGHT:
                    greenTank.rightRotate = False
                # purpleTank
                if event.key == pygame.K_e:
                    purpleTank.forwardDirection = False
                if event.key == pygame.K_d:
                    purpleTank.backwardDirection = False
                if event.key == pygame.K_s:
                    purpleTank.leftRotate = False
                if event.key == pygame.K_f:
                    purpleTank.rightRotate = False
                if event.type == GAME_GLOBALS.QUIT:
                    quitGame()

        surface.fill((0, 0, 0))
        text("Green Tank Score: " + str(greenTank.score), 291, 690, 100, 100, (255, 255, 255))
        text("Purple Tank Score: " + str(purpleTank.score), 1000, 690, 100, 100, (255, 255, 255))
        world.drawMap(surface)
        greenTank.move(surface)
        greenTank.rotate(surface)
        greenTank.drawTank(surface)
        purpleTank.move(surface)
        purpleTank.rotate(surface)
        purpleTank.drawTank(surface)
        for bullet in purpleTank.bullets:
            collisionKind = bullet.collision(surface)
            
            if collisionKind == "GREEN TANK COLLISION":
                greenTank.isWracked = True
                greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                purpleTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                purpleTank.isWracked = True
                purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                purpleTank.bullets.remove(bullet)
            bullet.draw(surface)
        for bullet in greenTank.bullets:
            collisionKind = bullet.collision(surface)
            if collisionKind == "GREEN TANK COLLISION":
                greenTank.isWracked = True
                greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                greenTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                purpleTank.isWracked = True
                purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                greenTank.bullets.remove(bullet)
            bullet.draw(surface)
        pygame.display.update()


def startmenu():
    startmenu = True
    while startmenu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        screensize = (1366, 768)
        surface.blit(bg, (0, 0))
        largetext = pygame.font.Font('freesansbold.ttf', 75)
        textsurface = largetext.render("LET'S BATTLE!", False, (255, 255, 255))
        text_welcome = textsurface.get_rect(center=(screensize[0] / 2, screensize[1] / 4))
        surface.blit(textsurface, text_welcome)
        button('PLAY', screensize[0] / 2 - 400, screensize[1] / 2, 800, 50, (78, 63, 177), (103, 177, 29), startedgame)
        button('QUIT', screensize[0] / 2 - 400, screensize[1] / 2 + 120, 800, 50, (78, 63, 177), (103, 177, 29),
               quitGame)
        pygame.display.update()


def pausemenu():
    pause = True
    while pause:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        screensize = (1366, 768)
        surface.blit(bg1, (0, 0))
        text = pygame.font.Font('freesansbold.ttf', 75)
        textsurface = text.render("PAUSE", False, (255, 255, 255))
        text1 = textsurface.get_rect(center=(screensize[0] / 2, screensize[1] / 4))
        surface.blit(textsurface, text1)
        button('RESUME', screensize[0] / 2 - 400, screensize[1] / 2, 800, 50, (250, 0, 0), (255, 255, 0), startedgame)
        button('QUIT', screensize[0] / 2 - 400, screensize[1] / 2 + 120, 800, 50, (250, 0, 0), (255, 255, 0), quitGame)
        pygame.display.update()


def button(msg, btn_x, btn_y, width, height, inactive_color, active_color, action=None):
    click = pygame.mouse.get_pressed()
    mouse = pygame.mouse.get_pos()
    if btn_x + width > mouse[0] > btn_x and btn_y + height > mouse[1] > btn_y:
        pygame.draw.rect(surface, active_color, (btn_x, btn_y, width, height))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(surface, inactive_color, (btn_x, btn_y, width, height))
        # surface.blit(menu, (0, 0))
    text = pygame.font.Font('freesansbold.ttf', 30)
    textsurf = text.render(msg, False, (0, 0, 0))
    text_welcome = textsurf.get_rect(center=(btn_x + width / 2, btn_y + height / 2))
    surface.blit(textsurf, text_welcome)


def text(msg, btn_x, btn_y, width, height,color):
    text = pygame.font.Font('freesansbold.ttf', 30)
    textsurf = text.render(msg, False, color)
    text_welcome = textsurf.get_rect(center=(btn_x + width / 2, btn_y + height / 2))
    surface.blit(textsurf, text_welcome)


startmenu()
