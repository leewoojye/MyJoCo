import pygame
import numpy as np
from numpy import sin, cos

# 환경 설정
WIDTH, HEIGHT = 800, 600
L1, L2 = 150, 150  # 막대기 길이
M1, M2 = 10.0, 10.0 # 질량
G = 9.8            # 중력

# 초기 상태 [theta1, theta2, d_theta1, d_theta2]
state = np.array([np.pi/2, np.pi/2, 0.0, 0.0])

def get_derivatives(state):
    t1, t2, w1, w2 = state
    
    # 이중 진자의 복잡한 가속도 수식 (라그랑주 결과물)
    # 분모 항
    den1 = (2*M1 + M2 - M2*cos(2*t1 - 2*t2))
    
    # theta1의 가속도 (dw1)
    dw1 = (-G*(2*M1 + M2)*sin(t1) - M2*G*sin(t1 - 2*t2) - 
           2*sin(t1 - t2)*M2*(w2**2*L2 + w1**2*L1*cos(t1 - t2))) / (L1 * den1)
    
    # theta2의 가속도 (dw2)
    dw2 = (2*sin(t1 - t2)*(w1**2*L1*(M1 + M2) + G*(M1 + M2)*cos(t1) + 
           w2**2*L2*M2*cos(t1 - t2))) / (L2 * den1)
    
    return np.array([w1, w2, dw1, dw2])

def rk4_step(state, dt):
    k1 = get_derivatives(state)
    k2 = get_derivatives(state + k1 * dt / 2)
    k3 = get_derivatives(state + k2 * dt / 2)
    k4 = get_derivatives(state + k3 * dt)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
dt = 0.05

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); exit()

    # 물리 업데이트 (RK4)
    state = rk4_step(state, dt)
    
    # 좌표 변환 (각도 -> 화면 좌표)
    x1 = WIDTH/2 + L1 * sin(state[0])
    y1 = 200 + L1 * cos(state[0])
    x2 = x1 + L2 * sin(state[1])
    y2 = y1 + L2 * cos(state[1])

    # 그리기
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0,0,0), (WIDTH/2, 200), (x1, y1), 5)
    pygame.draw.line(screen, (0,0,0), (x1, y1), (x2, y2), 5)
    pygame.draw.circle(screen, (0,128,255), (int(x1), int(y1)), 15)
    pygame.draw.circle(screen, (255,128,0), (int(x2), int(y2)), 15)
    
    pygame.display.flip()
    clock.tick(60)