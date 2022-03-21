use plotters::{coord::Shift, prelude::*};

use crate::Series;

pub(crate) struct GifRender<'a> {
    root: DrawingArea<BitMapBackend<'a>, Shift>,
}

impl<'a> GifRender<'a> {
    pub(crate) fn new(filename: &str, dims: (u32, u32)) -> Self {
        let root = BitMapBackend::gif(filename, dims, 100).unwrap().into_drawing_area();

        Self {
            root,
        }
    }

    pub(crate) fn update(
        &mut self,
        targets: &Series,
        train_preds: &Series,
        test_preds: &Series,
    ) {
        self.root.fill(&WHITE).unwrap();
        let ts_min = targets[0].0;
        let ts_max = targets[targets.len() - 1].0;
        let mut target_min: f64 = targets[0].1;
        let mut target_max: f64 = targets[targets.len() - 1].1;
        for t in targets {
            if t.1 < target_min {
                target_min = t.1;
            }
            if t.1 > target_max {
                target_max = t.1;
            }
        }
        info!("target_min: {}, target_max: {}", target_min, target_max);

        let areas = self.root.split_evenly((1, 1));

        let mut cc0 = ChartBuilder::on(&areas[0])
            .margin(5)
            .set_all_label_area_size(50)
            .caption("values", ("sans-serif", 30).into_font().with_color(&BLACK))
            .build_cartesian_2d(ts_min..ts_max, target_min..target_max)
            .unwrap();
        cc0.configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.4}", v))
            .draw()
            .unwrap();

        cc0.draw_series(LineSeries::new(targets.clone(), &BLACK))
            .unwrap()
            .label("targets")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
        cc0.draw_series(LineSeries::new(train_preds.clone(), &RED))
            .unwrap()
            .label("train_preds")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        cc0.draw_series(LineSeries::new(test_preds.clone(), &GREEN))
            .unwrap()
            .label("test_preds")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        cc0.configure_series_labels().border_style(&BLACK).draw().unwrap();

        self.root.present().unwrap();
    }
}
