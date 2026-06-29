# 可微渲染
| 202311081051 | 王婧怡 | 计算机科学与技术 |
| --- | --- | --- |


## <font style="color:rgb(15, 17, 21);">1. 实验目标</font>
<font style="color:rgb(15, 17, 21);">通过本实验，你将能够：</font>

+ <font style="color:rgb(15, 17, 21);">理解并掌握可微光栅化的原理，特别是在处理离散几何体（Mesh）边界时的数学近似方法</font>
+ <font style="color:rgb(15, 17, 21);">掌握如何通过多视角二维图像（剪影）反推并优化三维空间中的网格顶点坐标</font>
+ <font style="color:rgb(15, 17, 21);">深刻理解在网格优化过程中，正则化对于防止拓扑崩坏和陷入局部最优的决定性作用</font>



## <font style="color:rgb(15, 17, 21);">2. 实验原理</font>
### <font style="color:rgb(15, 17, 21);">2.1 核心思想</font>
<font style="color:rgb(15, 17, 21);">将一个初始的球体通过梯度下降，逐渐"捏"成一头奶牛的形状。这个过程需要解决两个核心问题：</font>



### <font style="color:rgb(15, 17, 21);">2.2 防梯度消失：软光栅化</font>
<font style="color:rgb(15, 17, 21);">在传统渲染（硬光栅化）中，像素要么在三角形内，要么在三角形外。这种阶跃变化导致边界处的梯度为 0（即发生梯度消失），网络无法知道顶点该往哪个方向移动。</font>

<font style="color:rgb(15, 17, 21);">解决方案：软光栅化</font>

<font style="color:rgb(15, 17, 21);">通过计算像素到三角形边缘的距离，并利用 Sigmoid 函数在边界处产生平滑的概率过渡：</font>

$$
A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right) 
$$

<font style="color:rgb(15, 17, 21);">其中</font><font style="color:rgb(15, 17, 21);"> </font>**<font style="color:rgb(15, 17, 21);">σ</font>**<font style="color:rgb(15, 17, 21);"> </font><font style="color:rgb(15, 17, 21);">控制边缘的模糊程度。这样，即使顶点在像素外部，也能提供微小但非零的梯度，引导顶点向正确方向移动。</font>



### <font style="color:rgb(15, 17, 21);">2.3 防局部最优：网格正则化</font>
<font style="color:rgb(15, 17, 21);">如果仅依靠图像差异去移动顶点，顶点会为了迎合某个视角的投影而疯狂交叉、重叠，最终变成一团充满尖刺的"刺猬"，彻底陷入局部最优。</font>

<font style="color:rgb(15, 17, 21);">解决方案：三种正则化损失</font>

| <font style="color:rgb(15, 17, 21);">正则化项</font> | <font style="color:rgb(15, 17, 21);">作用</font> |
| --- | --- |
| <font style="color:rgb(15, 17, 21);">拉普拉斯平滑</font> | <font style="color:rgb(15, 17, 21);">约束相邻顶点，防止表面出现尖锐突起</font> |
| <font style="color:rgb(15, 17, 21);">边长一致性</font> | <font style="color:rgb(15, 17, 21);">惩罚过长或过短的边，防止三角形严重拉伸</font> |
| <font style="color:rgb(15, 17, 21);">法线一致性</font> | <font style="color:rgb(15, 17, 21);">约束相邻三角形面的法线方向接近，保持表面平滑</font> |


<font style="color:rgb(15, 17, 21);">总损失函数：</font>

$$
L_{total} = L_{silhouette} + w_{lap}L_{lap} + w_{edge}L_{edge} + w_{normal}L_{normal} 
$$



## <font style="color:rgb(15, 17, 21);">3. 实验任务与步骤</font>
### <font style="color:rgb(15, 17, 21);">3.1 环境配置</font>
本实验依托魔塔平台完成需要安装以下依赖：</font>

```plain
# 1. 升级包管理工具
!pip install --upgrade pip

# 2. 安装前置依赖项与加速器（ninja 用于多核加速编译）
!pip install fvcore iopath matplotlib ninja

# 3. 使用 Gitee 链接直接源码编译安装
# ⚠️ 注意：加入 --no-build-isolation 可以防止某些云平台的严格沙箱隔离导致的编译报错
# 整个过程大约需要 5-10 分钟，请耐心等待！
!pip install "git+https://gitee.com/hongwenzhang/pytorch3d.git" --no-build-isolation
```



### <font style="color:rgb(15, 17, 21);">3.2 加载目标模型并生成参考图</font>
1. <font style="color:rgb(15, 17, 21);">载入目标网格：加载 </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">cow.obj</font>`<font style="color:rgb(15, 17, 21);"> 文件</font>
2. <font style="color:rgb(15, 17, 21);">归一化处理：将模型中心对齐到原点，并缩放到单位尺寸</font>
3. <font style="color:rgb(15, 17, 21);">多视角渲染：在空间中均匀设置 20 个摄像机视角，渲染出奶牛的目标剪影图</font>

```plain
# 核心代码示例
verts, faces, _ = load_obj(obj_path)
verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
cow_mesh = Meshes(verts=[verts], faces=[faces_idx])

# 多视角配置
num_views = 20
cameras = FoVPerspectiveCameras(
    device=device,
    R=look_at_view_transform(2.7, torch.zeros(num_views), 
                             torch.linspace(-180, 180, num_views))[0],
    T=look_at_view_transform(2.7, torch.zeros(num_views), 
                             torch.linspace(-180, 180, num_views))[1]
)
```

---

### <font style="color:rgb(15, 17, 21);">3.3 初始化源模型与渲染管线</font>
1. <font style="color:rgb(15, 17, 21);">初始化源网格：创建一个细分等级为 4 的二十面体球体</font>
2. <font style="color:rgb(15, 17, 21);">配置渲染管线：构建基于 PyTorch3D 的软剪影光栅化器</font>

```plain
# 球体初始化
src_mesh = ico_sphere(4, device)

# 可微渲染器配置
rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=RasterizationSettings(
        image_size=256,
        blur_radius=np.log(1./1e-4 - 1.)*1e-4,
        faces_per_pixel=50
    )
)
shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
```

---

### <font style="color:rgb(15, 17, 21);">3.4 执行可微优化循环</font>
#### <font style="color:rgb(15, 17, 21);">优化流程：</font>
1. <font style="color:rgb(15, 17, 21);">设置可微参数：将球体的顶点偏移量 </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">deform_verts</font>`<font style="color:rgb(15, 17, 21);"> 设为 </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">requires_grad=True</font>`
2. <font style="color:rgb(15, 17, 21);">前向传播</font><font style="color:rgb(15, 17, 21);">：计算当前形变后球体的剪影图</font>
3. <font style="color:rgb(15, 17, 21);">计算损失</font><font style="color:rgb(15, 17, 21);">：</font>
    - <font style="color:rgb(15, 17, 21);">剪影 MSE Loss</font>
    - <font style="color:rgb(15, 17, 21);">拉普拉斯平滑 Loss</font>
    - <font style="color:rgb(15, 17, 21);">边长一致性 Loss</font>
    - <font style="color:rgb(15, 17, 21);">法线一致性 Loss</font>
4. <font style="color:rgb(15, 17, 21);">反向传播：使用 SGD 优化器更新顶点偏移量</font>

```plain
# 核心优化循环
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

for i in range(epochs):
    optimizer.zero_grad()
    
    # 形变网格
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # 渲染剪影
    pred_silhouette = shader(rasterizer(new_src_mesh.extend(num_views)), 
                             new_src_mesh.extend(num_views))[..., 3]
    
    # 计算总损失
    loss_silhouette = ((pred_silhouette - target_silhouette) ** 2).mean()
    loss = loss_silhouette + \
           1.0 * mesh_laplacian_smoothing(new_src_mesh) + \
           0.1 * mesh_edge_loss(new_src_mesh) + \
           0.01 * mesh_normal_consistency(new_src_mesh)
    
    loss.backward()
    optimizer.step()
```

---

### <font style="color:rgb(15, 17, 21);">3.5 可视化与输出</font>
+ <font style="color:rgb(15, 17, 21);">实时监控：每 20 步（300 次迭代）或每 50 步（500 次迭代）展示一次优化进度</font>
+ <font style="color:rgb(15, 17, 21);">模型保存</font><font style="color:rgb(15, 17, 21);">：定期保存中间状态的</font><font style="color:rgb(15, 17, 21);"> </font>`<font style="color:rgb(15, 17, 21);background-color:rgb(235, 238, 242);">.obj</font>`<font style="color:rgb(15, 17, 21);"> </font><font style="color:rgb(15, 17, 21);">文件</font>
+ <font style="color:rgb(15, 17, 21);">最终输出：优化完成后，输出最终模型</font>

<font style="color:rgb(15, 17, 21);">python</font>

```plain
# 保存模型
save_path = os.path.join(output_dir, f"mesh_epoch_{i:03d}.obj")
save_obj(save_path, current_verts, current_faces)

# 可视化对比
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(target_silhouette[0].cpu().numpy(), cmap='gray')
ax[0].set_title("Ground Truth Silhouette")
ax[0].axis("off")
ax[1].imshow(pred_silhouette[0].detach().cpu().numpy(), cmap='gray')
ax[1].set_title(f"Optimizing... (Epoch {i})")
ax[1].axis("off")
plt.show()
```

---

## <font style="color:rgb(15, 17, 21);">4. 实验结果</font>
### <font style="color:rgb(15, 17, 21);">4.1 实验设置</font>
<font style="color:rgb(15, 17, 21);">本次实验共进行了两组对比测试：</font>

| <font style="color:rgb(15, 17, 21);">实验组</font> | <font style="color:rgb(15, 17, 21);">迭代次数</font> | <font style="color:rgb(15, 17, 21);">保存频率</font> | <font style="color:rgb(15, 17, 21);">输出目录</font> | <font style="color:rgb(15, 17, 21);">备注</font> |
| --- | --- | --- | --- | --- |
| <font style="color:rgb(15, 17, 21);">实验 1</font> | <font style="color:rgb(15, 17, 21);">300</font> | <font style="color:rgb(15, 17, 21);">每 20 步</font> | output_meshes/ | <font style="color:rgb(15, 17, 21);">共生成 16 个中间模型</font> |
| <font style="color:rgb(15, 17, 21);">实验 2</font> | <font style="color:rgb(15, 17, 21);">500</font> | <font style="color:rgb(15, 17, 21);">每 50 步</font> | output_meshes1/ | <font style="color:rgb(15, 17, 21);">共生成 11 个中间模型</font> |


### <font style="color:rgb(15, 17, 21);">4.2 最优效果展示</font>
#### <font style="color:rgb(15, 17, 21);">实验 1：300 次迭代</font>
<img width="1269" height="705" alt="d0a97a46b172cbf69f2766c33dd6be79" src="https://github.com/user-attachments/assets/c53f249f-ca69-462d-aa2a-33dcfe524efb" />
<img width="1004" height="923" alt="299" src="https://github.com/user-attachments/assets/b888e2be-063a-4659-bc3e-218a1bb0d2a7" />

#### <font style="color:rgb(15, 17, 21);">实验 2：500 次迭代</font>
<img width="1248" height="756" alt="f948218c334bc8615ddcab644c6f2d0d" src="https://github.com/user-attachments/assets/e4f3b414-35f6-4d51-b0f5-f251c0c0eb86" />
<img width="1007" height="972" alt="499" src="https://github.com/user-attachments/assets/e06d8ace-dfaa-473a-a1bc-1e7a52611695" />
