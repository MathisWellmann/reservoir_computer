mod gif_render;
mod gif_render_optimizer;
mod plot;

pub use gif_render::GifRender;
pub use gif_render_optimizer::GifRenderOptimizer;
pub use plot::plot;

pub type Series = Vec<(f64, f64)>;
