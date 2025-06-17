use wgpu::{Device, Queue, Surface, SurfaceConfiguration};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

pub struct App<'window> {
    surface: Option<Surface<'window>>,
    device: Option<Device>,
    queue: Option<Queue>,
    config: Option<SurfaceConfiguration>,
    window: Option<Window>,
}

impl<'window> App<'window> {
    pub fn new() -> Self {
        Self {
            surface: None,
            device: None,
            queue: None,
            config: None,
            window: None,
        }
    }

    pub fn run_app(
        mut self,
        event_loop: winit::event_loop::EventLoop<()>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    fn init_wgpu(&mut self, window: &'window Window) -> Result<(), Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(window) }?;

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

    fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(surface), (Some(device)), (Some(queue)), (Some(config))) =
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

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(
            WindowAttributes::default()
                .with_title("ASCII Robot")
                .with_inner_size(winit::dpi::LogicalSize::new(640, 480)),
        ) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Window creation failed: {}", e);
                event_loop.exit();
                return;
            }
        };

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
                if let (Some(surface), (Some(device)), (Some(config))) =
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
