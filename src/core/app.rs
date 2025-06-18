use image::DynamicImage;
use nokhwa::{
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::sync::{Arc, Mutex};
use wgpu::{Device, Queue, Surface, SurfaceConfiguration, TextureFormat};
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
        }))
        .unwrap();

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                label: None,
                memory_hints: Default::default(),
                trace: Default::default(),
            }))?;

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
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
        // List available cameras
        let camers = query(ApiBackend::Auto)?;
        if camers.is_empty() {
            return Err("No camera found".into());
        }

        // Use the first available camera
        println!("Found {} camers - Select camera with index 0", camers.len());
        let index = CameraIndex::Index(0);
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        // Create camera with explicit backend
        let camera = Camera::new(index, format)?;

        self.camera = Some(Arc::new(Mutex::new(camera)));
        Ok(())
    }

    fn update_camera_texture(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(device), Some(queue), Some(camera)) = (&self.device, &self.queue, &self.camera)
        {
            let mut camera = camera.lock().unwrap();

            // Capture a frame from the camera
            if let Ok(frame) = camera.frame() {
                let frame_buffer = frame.decode_image::<RgbFormat>()?;
                let image = DynamicImage::ImageRgb8(frame_buffer);

                let width = image.width();
                let height = image.height();
                let rgb = image.to_rgb8();

                // Create or update the texture
                if self.camera_texture.is_none() {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Camera Texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });

                    self.camera_texture = Some(texture);
                    self.camera_texture_view = Some(
                        self.camera_texture
                            .as_ref()
                            .unwrap()
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                    );
                }

                // Update the texture with new frame data
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: self.camera_texture.as_ref().unwrap(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgb,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(3 * width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
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

        if let (Some(surface), Some(device), Some(queue), Some(_config)) =
            (&self.surface, &self.device, &self.queue, &self.config)
        {
            let frame = surface.get_current_texture()?;
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

            {
                let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
            }

            queue.submit(std::iter::once(encoder.finish()));
            frame.present();
        }
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
            // Don't exit - run without camera
            eprintln!("Running without camera support");
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
        self.surface = None;
        self.window = None;
    }
}
