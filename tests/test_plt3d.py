import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 테스트용 점 하나 찍기
ax.scatter(0, 0, 0, color='red', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title("Matplotlib 3D Test")
plt.show()