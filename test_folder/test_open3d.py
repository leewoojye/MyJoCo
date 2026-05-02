import open3d as o3d

# 1. 다운로드한 stl 파일 이름 (예: 'link1.stl')
# 본인이 다운받은 파일 이름과 경로로 바꿔주세요.
mesh_file_path = "/Users/woojyelee/workspace/my_robotics/robotis_mujoco_menagerie/robotis_ffw/assets/ffw_b/arm_l_link1.stl" 

# 2. Open3D를 이용해 메쉬 읽어오기
print(f"[{mesh_file_path}] 파일을 불러오는 중...")
mesh = o3d.io.read_triangle_mesh(mesh_file_path)

# 3. 로드 성공 여부 확인
if not mesh.has_triangles():
    print("메쉬를 불러오지 못했습니다. 파일 경로와 포맷을 확인해주세요.")
else:
    # 4. 시각적 품질 향상을 위해 법선(Normal) 계산 (필수)
    mesh.compute_vertex_normals()
    
    # 5. 원하는 색상 입히기 [R, G, B] (0.0 ~ 1.0 사이 값)
    # 아래는 약간의 회색빛을 주는 예시입니다.
    mesh.paint_uniform_color([0.6, 0.6, 0.6])
    
    # 6. 화면에 띄우기
    print("마우스 왼쪽 클릭: 회전 / 마우스 휠: 확대, 축소")
    o3d.visualization.draw_geometries([mesh], window_name="STL Viewer (Open3D)")