import os
import sys
import gmsh


def generate_die_bonding_mesh(
    die_width=2.0,  # Die 寬度 (mm)
    die_height=0.3,  # Die 厚度 (mm)
    epoxy_thick=0.03,  # 膠層厚度 (mm)
    sub_width=4.0,  # 基板寬度 (mm)
    sub_height=0.5,  # 基板厚度 (mm)
    mesh_size_coarse=0.2,  # 遠端粗網格尺寸
    mesh_size_fine=0.01,  # 關鍵介面細網格尺寸
    output_filename="die_bonding.vtk",
):
    """使用 Gmsh API 進行參數化建模與剖網格，並匯出 VTK 格式。"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # 顯示 Gmsh 執行 Log
    gmsh.model.add("die_bonding_model")

    # ---------------------------------------------------------
    # 1. 使用 OpenCASCADE 引擎宣告 3D 幾何實體 (Volumes)
    # ---------------------------------------------------------
    occ = gmsh.model.occ

    # Substrate (基板)
    sub_tag = occ.addBox(
        -sub_width / 2.0, -sub_width / 2.0, -sub_height, sub_width, sub_width, sub_height
    )

    # Epoxy (膠層)
    epoxy_tag = occ.addBox(
        -die_width / 2.0, -die_width / 2.0, 0.0, die_width, die_width, epoxy_thick
    )

    # Die (晶粒)
    die_tag = occ.addBox(
        -die_width / 2.0,
        -die_width / 2.0,
        epoxy_thick,
        die_width,
        die_width,
        die_height,
    )

    # 執行布林碎片運算 (Fragment)，確保交界面的網格節點能完美印合 (Conformal Mesh)
    synced_out, _ = occ.fragment([(3, sub_tag), (3, epoxy_tag), (3, die_tag)], [])
    occ.synchronize()

    # ---------------------------------------------------------
    # 2. 設置網格尺寸控制 (Mesh Size Fields / Refinement)
    # ---------------------------------------------------------
    # 設定全域基礎網格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_fine)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_coarse)

    # 針對 Epoxy 區域進行局部網格加密 (Distance Field + Threshold Field)
    # 找出與 Epoxy 相關的幾何點
    epoxy_entities = [tag for dim, tag in synced_out if dim == 3]

    # 利用 Box Field 在膠層與 Die 貼合區精細剖網格
    field_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field_box, "VIn", mesh_size_fine)
    gmsh.model.mesh.field.setNumber(field_box, "VOut", mesh_size_coarse)
    gmsh.model.mesh.field.setNumber(field_box, "XMin", -die_width / 2.0 - 0.1)
    gmsh.model.mesh.field.setNumber(field_box, "XMax", die_width / 2.0 + 0.1)
    gmsh.model.mesh.field.setNumber(field_box, "YMin", -die_width / 2.0 - 0.1)
    gmsh.model.mesh.field.setNumber(field_box, "YMax", die_width / 2.0 + 0.1)
    gmsh.model.mesh.field.setNumber(field_box, "ZMin", -0.05)
    gmsh.model.mesh.field.setNumber(field_box, "ZMax", epoxy_thick + 0.05)

    gmsh.model.mesh.field.setAsBackgroundMesh(field_box)

    # ---------------------------------------------------------
    # 3. 產生 3D 體積網格並匯出
    # ---------------------------------------------------------
    gmsh.model.mesh.generate(3)
    gmsh.write(output_filename)
    gmsh.finalize()
    print(f"[Done] 網格已成功建立並儲存至 {output_filename}")


def generate_mesh(
    output_filename="die_bonding_mesh.msh",
    gap_length=2.0,  # 膠層截面長度 L (mm)
    gap_height=0.05,  # 膠層間隙高度 H (mm)
    mesh_size=0.01,  # 網格尺寸 (mm)
):
    """產生 02_solve_simulation.py 使用的 2D 膠層截面網格 (.msh)。

    Physical tags (與求解器約定一致):
      - Surface 1 : Epoxy 膠層區域
      - Curve 1   : 底部基板介面 (固定、加熱)
      - Curve 2   : 頂部 Die 介面 (Bond Head 下壓)
      - Curve 3   : 左右側出口 (大氣壓力釋放)
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("epoxy_gap_2d")

    occ = gmsh.model.occ
    surf = occ.addRectangle(-gap_length / 2.0, 0.0, 0.0, gap_length, gap_height)
    occ.synchronize()

    bottom, top, sides = [], [], []
    for _, tag in gmsh.model.getBoundary([(2, surf)], oriented=False):
        x, y, _ = gmsh.model.occ.getCenterOfMass(1, tag)
        if abs(y) < 1e-9:
            bottom.append(tag)
        elif abs(y - gap_height) < 1e-9:
            top.append(tag)
        else:
            sides.append(tag)

    gmsh.model.addPhysicalGroup(2, [surf], 1, name="epoxy")
    gmsh.model.addPhysicalGroup(1, bottom, 1, name="substrate_interface")
    gmsh.model.addPhysicalGroup(1, top, 2, name="die_interface")
    gmsh.model.addPhysicalGroup(1, sides, 3, name="lateral_outlets")

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()
    print(f"[Done] 2D 膠層網格已儲存至 {output_filename}")


def visualize_mesh_with_pyvista(vtk_file="die_bonding.vtk"):
    """使用 PyVista 讀取並進行 3D 互動視覺化與剖面觀察。"""
    import pyvista as pv  # 延遲匯入: DOLFINx 容器內未安裝 PyVista

    if not os.path.exists(vtk_file):
        raise FileNotFoundError(f"找不到檔案: {vtk_file}")

    # 讀取 Gmsh 導出的 VTK 網格數據
    mesh = pv.read(vtk_file)

    print("\n--- 網格資訊 ---")
    print(f"節點數量 (Points): {mesh.n_points}")
    print(f"單元數量 (Cells):  {mesh.n_cells}")

    # 設定 PyVista 畫布
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.set_background("#1e1e24")

    # 繪製主體網格 (顯示網格邊線 Wireframe)
    plotter.add_mesh(
        mesh,
        show_edges=True,
        edge_color="#444444",
        color="#83c5be",
        opacity=0.6,
        label="Die Bonding Geometry",
    )

    # 增加一個動態 Y-Z 剖面工具 (Clip Plane)，方便觀察內部 Epoxy 與 Die 介面的網格細化程度
    plotter.add_mesh_clip_plane(
        mesh,
        normal="x",
        invert=False,
        widget_color="#e63946",
        show_edges=True,
        color="#ffddd2",
    )

    plotter.add_axes()
    plotter.add_title("Die Bonding 3D Parameterized Mesh (Gmsh + PyVista)")
    plotter.show()


if __name__ == "__main__":
    # 步驟 1: 設定參數並生成 3D 網格
    mesh_file = "die_bonding_structure.vtk"
    generate_die_bonding_mesh(
        die_width=2.5,  # 可自由調整 Die 尺寸 (mm)
        die_height=0.35,
        epoxy_thick=0.025,  # 膠層極薄 (25 μm)
        sub_width=5.0,
        mesh_size_coarse=0.3,
        mesh_size_fine=0.015,  # 微米級極細區域網格
        output_filename=mesh_file,
    )

    # 步驟 2: 啟動 PyVista 進行 3D 網格檢測 (無顯示環境時略過, 例如 Docker 容器)
    if sys.platform == "darwin" or os.environ.get("DISPLAY"):
        visualize_mesh_with_pyvista(mesh_file)
    else:
        print("[Skip] 無顯示環境, 略過 PyVista 視覺化")