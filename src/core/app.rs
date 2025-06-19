use image::DynamicImage;
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{
        ApiBackend, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use std::sync::{Arc, Mutex};
use wgpu::{
    AddressMode, BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, Extent3d, FilterMode, FragmentState, LoadOp,
    MultisampleState, Operations, Origin3d, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderStages, Surface, SurfaceConfiguration, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

pub struct App {
    surface: Option<Surface<'static>>,
    device: Option<Device>,
    queue: Option<Queue>,
    config: Option<SurfaceConfiguration>,
    window: Option<Window>,
    camera: Option<Arc<Mutex<Camera>>>,
    camera_texture: Option<wgpu::Texture>,
    camera_texture_view: Option<wgpu::TextureView>,
    camera_sampler: Option<Sampler>,
    camera_bind_group: Option<BindGroup>,
    camera_bind_group_layout: Option<BindGroupLayout>,
    camera_pipeline: Option<RenderPipeline>,
}

impl App {
    pub fn new() -> Self {
        Self {
            surface: None,
            device: None,
            queue: None,
            config: None,
            window: None,
            camera: None,
            camera_texture: None,
            camera_texture_view: None,
            camera_sampler: None,
            camera_bind_group: None,
            camera_bind_group_layout: None,
            camera_pipeline: None,
        }
    }

    fn init_wgpu(&mut self, window: &Window) -> Result<(), Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let surface = unsafe {
            std::mem::transmute::<Surface<'_>, Surface<'static>>(instance.create_surface(window)?)
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;

        // Get GPU limits
        let limits = adapter.limits();
        let max_texture_dimension = limits.max_texture_dimension_2d;
        println!("GPU max texture dimension: {}", max_texture_dimension);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                label: None,
                memory_hints: Default::default(),
                trace: Default::default(),
            }))?;

        // Get window size and clamp to GPU limits
        let mut size = window.inner_size();
        println!("Requested window size: {}x{}", size.width, size.height);

        size.width = size.width.clamp(1, max_texture_dimension);
        size.height = size.height.clamp(1, max_texture_dimension);
        println!("Clamped window size: {}x{}", size.width, size.height);

        let surface_caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_caps.formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);

        Ok(())
    }

    fn init_camera(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let cameras = query(ApiBackend::Auto)?;
        if cameras.is_empty() {
            return Err("No camera found".into());
        }

        println!("Found {} cameras - Using first camera", cameras.len());
        let index = CameraIndex::Index(0);

        // Get camera info
        let camera_info = &cameras[0];
        println!("Camera name: {}", camera_info.human_name());

        // Try to find a compatible format
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);

        let mut camera = Camera::new(index, format)?;

        // Set to 720p resolution
        let resolution = Resolution::new(1280, 720);
        camera.set_resolution(resolution)?;

        camera.open_stream()?;

        self.camera = Some(Arc::new(Mutex::new(camera)));
        Ok(())
    }

    fn create_render_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let device = self.device.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();

        // Create shader module
        let shader_source = r#"
    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) uv: vec2<f32>,
    }

    @vertex
    fn vs_main(@builtin(vertex_index) vert_idx: u32) -> VertexOutput {
        let pos = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, -1.0),  // Bottom-left
            vec2<f32>(-1.0, 1.0),   // Top-left
            vec2<f32>(1.0, -1.0),   // Bottom-right
            vec2<f32>(1.0, 1.0)     // Top-right
        );
        
        let uv = array<vec2<f32>, 4>(
            vec2<f32>(0.0, 1.0),  // Bottom-left
            vec2<f32>(0.0, 0.0),  // Top-left
            vec2<f32>(1.0, 1.0),  // Bottom-right
            vec2<f32>(1.0, 0.0)   // Top-right
        );
        
        var output: VertexOutput;
        output.position = vec4<f32>(pos[vert_idx], 0.0, 1.0);
        output.uv = uv[vert_idx];
        return output;
    }

    @group(0) @binding(0)
    var camera_tex: texture_2d<f32>;

    @group(0) @binding(1)
    var camera_sampler: sampler;

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
        return textureSample(camera_tex, camera_sampler, input.uv);
    }
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Camera Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Camera Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Camera Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create sampler
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Camera Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        self.camera_sampler = Some(sampler);
        self.camera_pipeline = Some(pipeline);
        self.camera_bind_group_layout = Some(bind_group_layout);

        Ok(())
    }

    fn update_camera_texture(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (device, queue) = match (&self.device, &self.queue) {
            (Some(d), Some(q)) => (d, q),
            _ => return Ok(()),
        };

        let camera = match &self.camera {
            Some(cam) => cam,
            None => return Ok(()),
        };

        let mut camera = camera.lock().unwrap();

        // Get camera settings
        let resolution = camera.resolution();
        let width = resolution.width() as u32;
        let height = resolution.height() as u32;
        let fmt = camera.camera_format().format();

        println!("Camera format: {:?}, resolution: {}x{}", fmt, width, height);

        if let Ok(frame) = camera.frame() {
            println!("Frame buffer size: {}", frame.buffer().len());

            // Convert to RGBA based on the frame format
            let rgba = match fmt {
                FrameFormat::NV12 => {
                    // Handle NV12 format specifically
                    let buffer = frame.buffer();
                    let mut rgba = vec![0u8; (width * height * 4) as usize];

                    // For NV12, stride is typically width
                    let stride = width as usize;

                    println!(
                        "NV12 frame: width={}, height={}, stride={}, buffer_len={}",
                        width,
                        height,
                        stride,
                        buffer.len()
                    );

                    Self::nv12_to_rgba(buffer, &mut rgba, width as usize, height as usize, stride);
                    rgba
                }
                _ => {
                    // Try to decode other formats
                    if let Ok(decoded) = frame.decode_image::<RgbFormat>() {
                        // Convert to DynamicImage then to RGBA
                        DynamicImage::ImageRgb8(decoded).to_rgba8().to_vec()
                    } else {
                        eprintln!("Failed to decode frame format: {:?}", fmt);
                        return Ok(());
                    }
                }
            };

            // Create or update texture
            if self.camera_texture.is_none()
                || self.camera_texture.as_ref().unwrap().width() != width
                || self.camera_texture.as_ref().unwrap().height() != height
            {
                let texture = device.create_texture(&TextureDescriptor {
                    label: Some("Camera Texture"),
                    size: Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                let view = texture.create_view(&TextureViewDescriptor::default());
                self.camera_texture = Some(texture);
                self.camera_texture_view = Some(view);

                // Recreate bind group
                if let (Some(view), Some(layout), Some(sampler)) = (
                    &self.camera_texture_view,
                    &self.camera_bind_group_layout,
                    &self.camera_sampler,
                ) {
                    self.camera_bind_group =
                        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Camera Bind Group"),
                            layout,
                            entries: &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: BindingResource::TextureView(view),
                                },
                                BindGroupEntry {
                                    binding: 1,
                                    resource: BindingResource::Sampler(sampler),
                                },
                            ],
                        }));
                }
            }

            // Update texture
            if let Some(texture) = &self.camera_texture {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &rgba,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width),
                        rows_per_image: Some(height),
                    },
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        } else {
            eprintln!("Failed to capture frame from camera");
        }
        Ok(())
    }

    fn nv12_to_rgba(nv12: &[u8], rgba: &mut [u8], width: usize, height: usize, stride: usize) {
        let y_plane_size = stride * height;
        let uv_plane_size = stride * height / 2;

        if nv12.len() < y_plane_size + uv_plane_size {
            eprintln!(
                "NV12 buffer too small: expected {}, got {}",
                y_plane_size + uv_plane_size,
                nv12.len()
            );
            return;
        }

        let y_plane = &nv12[0..y_plane_size];
        let uv_plane = &nv12[y_plane_size..y_plane_size + uv_plane_size];

        for y in 0..height {
            for x in 0..width {
                let y_index = y * stride + x;
                let uv_index = (y / 2) * (stride / 2) + (x / 2);

                let y_val = y_plane[y_index] as f32;
                let u_val = uv_plane[uv_index] as f32 - 128.0;
                let v_val = uv_plane[uv_index + 1] as f32 - 128.0;

                // Use standard BT.709 conversion coefficients
                let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                let rgba_index = (y * width + x) * 4;
                rgba[rgba_index] = r;
                rgba[rgba_index + 1] = g;
                rgba[rgba_index + 2] = b;
                rgba[rgba_index + 3] = 255; // Alpha
            }
        }
    }

    fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Update camera texture with the latest frame
        if let Err(e) = self.update_camera_texture() {
            eprintln!("Camera update failed: {}", e);
        }

        let (surface, device, queue, _config) =
            match (&self.surface, &self.device, &self.queue, &self.config) {
                (Some(s), Some(d), Some(q), Some(c)) => (s, d, q, c),
                _ => return Ok(()),
            };

        let frame = surface.get_current_texture()?;
        let view = frame.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw camera texture if available
            if let (Some(pipeline), Some(bind_group)) =
                (&self.camera_pipeline, &self.camera_bind_group)
            {
                rpass.set_pipeline(pipeline);
                rpass.set_bind_group(0, bind_group, &[]);
                rpass.draw(0..4, 0..1); // Draw quad
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if let (Some(surface), Some(device), Some(config)) =
            (&self.surface, &self.device, &mut self.config)
        {
            // Get GPU limits
            let max_texture_dimension = device.limits().max_texture_dimension_2d;

            // Clamp new size to GPU limits
            let width = new_size.width.clamp(1, max_texture_dimension);
            let height = new_size.height.clamp(1, max_texture_dimension);

            if width != config.width || height != config.height {
                println!(
                    "Resizing to: {}x{} (max: {})",
                    width, height, max_texture_dimension
                );
                config.width = width;
                config.height = height;
                surface.configure(device, config);
            }

            // Request redraw after resize
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window at 1280x720 resolution
        let window = match event_loop.create_window(
            WindowAttributes::default()
                .with_title("Camera Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Window creation failed: {}", e);
                event_loop.exit();
                return;
            }
        };

        // Initialize WGPU
        if let Err(e) = self.init_wgpu(&window) {
            eprintln!("Failed to initialize WGPU: {}", e);
            event_loop.exit();
            return;
        }

        // Initialize camera
        if let Err(e) = self.init_camera() {
            eprintln!("Failed to initialize camera: {}", e);
        }

        // Create rendering pipeline
        if let Err(e) = self.create_render_pipeline() {
            eprintln!("Failed to create render pipeline: {}", e);
            event_loop.exit();
            return;
        }

        // Store window AFTER successful initialization
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Only handle events for our window
        if self.window.as_ref().map(|w| w.id()) != Some(window_id) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render() {
                    eprintln!("Render error: {}", e);
                    event_loop.exit();
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(new_size) => {
                self.handle_resize(new_size);
            }
            WindowEvent::ScaleFactorChanged {
                inner_size_writer: _,
                scale_factor: _,
            } => {
                if let Some(window) = &self.window {
                    let new_size = window.inner_size();
                    self.handle_resize(new_size);
                }
            }
            _ => {}
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(camera) = &self.camera {
            let mut cam = camera.lock().unwrap();
            let _ = cam.stop_stream();
        }
    }
}
