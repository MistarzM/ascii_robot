use image::DynamicImage;
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
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

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                label: None,
                memory_hints: Default::default(),
                trace: Default::default(),
            }))?;

        let size = window.inner_size();
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
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut camera = Camera::new(index, format)?;
        camera.open_stream()?; // Changed to open_stream()

        self.camera = Some(Arc::new(Mutex::new(camera)));
        Ok(())
    }

    fn create_render_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let device = self.device.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();

        // Create shader module
        let shader_source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) vert_idx: u32) -> @builtin(position) vec4<f32> {
                let pos = array<vec2<f32>, 4>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(-1.0, 1.0),
                    vec2<f32>(1.0, -1.0),
                    vec2<f32>(1.0, 1.0)
                );
                return vec4<f32>(pos[vert_idx], 0.0, 1.0);
            }

            @group(0) @binding(0)
            var camera_tex: texture_2d<f32>;

            @group(0) @binding(1)
            var camera_sampler: sampler;

            @fragment
            fn fs_main(@builtin(position) frag_coord: vec4<f32>) 
                -> @location(0) vec4<f32> {
                let uv = vec2<f32>(frag_coord.x / 640.0, frag_coord.y / 480.0);
                return textureSample(camera_tex, camera_sampler, uv);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            // Removed &
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
                entry_point: Some("vs_main"), // Removed Some()
                buffers: &[],
                compilation_options: Default::default(), // Added
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"), // Removed Some()
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(), // Added
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None, // Added
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
        self.camera_bind_group_layout = Some(bind_group_layout); // Store layout

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

        if let Ok(frame) = camera.frame() {
            let frame_buffer = frame.decode_image::<RgbFormat>()?;
            let image = DynamicImage::ImageRgb8(frame_buffer);
            let width = image.width();
            let height = image.height();

            // Convert to RGBA format
            let rgba = image.to_rgba8();

            // Create texture if it doesn't exist or if size changed
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

                // Create bind group if we have all components
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

            // Update texture with new frame data
            if let Some(texture) = &self.camera_texture {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        // Updated type
                        texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &rgba,
                    TexelCopyBufferLayout {
                        // Updated type
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
        }
        Ok(())
    }

    fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Update camera texture with the latest frame
        if let Err(e) = self.update_camera_texture() {
            eprintln!("Camera update failed: {}", e);
        }

        let (surface, device, queue, _config) = match (
            // Added _ to config
            &self.surface,
            &self.device,
            &self.queue,
            &self.config,
        ) {
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(
            WindowAttributes::default()
                .with_title("Camera Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(640, 480)),
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
        _window_id: WindowId,
        event: WindowEvent,
    ) {
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
                if let (Some(surface), Some(device), Some(config)) =
                    (&self.surface, &self.device, &mut self.config)
                {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(device, config);
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
