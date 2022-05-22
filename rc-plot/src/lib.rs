#[macro_use]
extern crate log;

mod gif_render;
mod gif_render_optimizer;
pub mod plot;
mod plot_gather;

pub use gif_render::GifRender;
pub use gif_render_optimizer::GifRenderOptimizer;
pub use plot::plot;
pub use plot_gather::PlotGather;

pub type Series = Vec<(f64, f64)>;
