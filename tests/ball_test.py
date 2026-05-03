import pygame
import sys

# 1. 환경 설정
WIDTH, HEIGHT = 800, 600
FPS = 60
dt = 1 / FPS  # 한 프레임당 시간 간격

# 물리 파라미터
GRAVITY = 980.0    # 중력 가속도 (픽셀 단위라 숫자가 큼)
ELASTICITY = 0.8   # 탄성 계수 (바닥에 닿을 때 에너지를 20% 잃음)

# 2. 공의 상태 (State)
ball_pos = [400, 50]  # [x, y]
ball_vel = [100, 0]   # [vx, vy]
ball_radius = 20

# 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 3. 시뮬레이션 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- 물리 연산 (The Core) ---
    # 가속도 -> 속도 업데이트
    ball_vel[1] += GRAVITY * dt
    
    # 속도 -> 위치 업데이트
    ball_pos[0] += ball_vel[0] * dt
    ball_pos[1] += ball_vel[1] * dt

    # 바닥 충돌 검사 (Collision Detection & Response)
    if ball_pos[1] + ball_radius > HEIGHT:
        ball_pos[1] = HEIGHT - ball_radius  # 바닥에 박히지 않게 보정
        ball_vel[1] *= -ELASTICITY           # 속도 반전 및 에너지 감소

    # 벽 충돌 검사
    if ball_pos[0] + ball_radius > WIDTH or ball_pos[0] - ball_radius < 0:
        ball_vel[0] *= -ELASTICITY

    # --- 그리기 ---
    screen.fill((255, 255, 255)) # 배경 흰색
    pygame.draw.circle(screen, (0, 128, 255), (int(ball_pos[0]), int(ball_pos[1])), ball_radius)
    pygame.display.flip()
    
    clock.tick(FPS)